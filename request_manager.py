#!/usr/bin/env python3
"""Centralized TPU VM request/reclaim manager.

This manager owns only GCP TPU VM entities: name, zone, type, and state. It does
not allocate or register xibo aliases. Existing xibo/MONITOR code can later bind
an idle VM name to an alias when it actually launches a job.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import logging
import os
import random
import re
import secrets
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on host image.
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = "/kmh-nfs-ssd-us-mount/code/qiao/work"
TPU_DLS_DIR = os.path.join(WORK_ROOT, "tpu_dls")
AUDIT_CACHE = os.path.join(TPU_DLS_DIR, ".tpu_audit_records.json")
WRAP_MASTER = os.path.join(TPU_DLS_DIR, "wrap_master.py")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "request_demand.yaml")
DEFAULT_STATE = os.path.join(BASE_DIR, "request_state.json")
DEFAULT_EVENTS = os.path.join(BASE_DIR, "events.jsonl")
DEFAULT_LOG = os.path.join(BASE_DIR, "request_manager.log")
DEFAULT_LOCK = os.path.join(BASE_DIR, "request_manager.lock")
VM_TXT = os.path.join(WORK_ROOT, "vm.txt")
TPU_LOCK_DIR = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"

PROJECT = "he-vision-group"
DEFAULT_REGIONS = ["us-central1", "us-east5", "asia-northeast1-b"]
KNOWN_REGION_ZONES = {
    "us-central1": ["us-central1-a", "us-central1-b"],
    "us-east5": ["us-east5-a", "us-east5-b"],
    "asia-northeast1": ["asia-northeast1-b"],
    "asia-northeast1-b": ["asia-northeast1-b"],
}
REGION_SA_MAP = {
    "us-central1": "bucket-us-central1@he-vision-group.iam.gserviceaccount.com",
    "us-east5": "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
    "asia-northeast1": "bucket-asia@he-vision-group.iam.gserviceaccount.com",
}
VERSION_BY_FAMILY = {
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}

TYPE_RE = re.compile(r"^(v[0-9]+[a-z]?)-([0-9]+)$")
NAME_TYPE_RE = re.compile(r"kmh-tpuvm-(v[0-9]+[a-z]?-[0-9]+)")
ZONE_RE = re.compile(r"^[a-z]+-[a-z0-9-]+-[a-z]$")
LOCK_TIME_FMT = "%Y-%m-%d_%H-%M-%S"
STATE_VERSION = 1


@dataclass(frozen=True)
class Demand:
    tpu_type: str
    target_idle: int
    max_inflight: int
    zones: Tuple[str, ...]


@dataclass(frozen=True)
class ManagerConfig:
    path: str
    enabled: bool
    dry_run: bool
    owner: str
    base_prefix: str
    regions: Tuple[str, ...]
    demands: Dict[str, Demand]
    loop_interval_seconds: int
    max_cache_age_seconds: int
    max_create_per_loop: int
    create_timeout_seconds: int
    create_workers: int
    min_attempt_interval_seconds: int
    new_vm_grace_seconds: int
    append_vm_txt: bool
    labels: Dict[str, str]
    cooldown_seconds: Dict[str, int]
    max_cooldown_seconds: int
    reclaim: Dict[str, Any]


class ConfigError(ValueError):
    pass


class AuditCacheError(RuntimeError):
    pass


def utc_now() -> float:
    return time.time()


def iso_ts(epoch: Optional[float] = None) -> str:
    if epoch is None:
        epoch = utc_now()
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat(timespec="seconds")


def setup_logging(log_path: str, verbose: bool = False) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(level)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read().strip()
        if not content:
            return default
        return json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read JSON file {path}: {exc}") from exc


def atomic_write_json(path: str, payload: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=indent, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", utc_now())
    payload.setdefault("ts_iso", iso_ts(payload["ts"]))
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def default_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "idle_observations": {},
        "managed_vms": {},
        "attempts": {},
        "cooldowns": {},
        "lifecycle_cache": {},
        "last_loop": None,
    }


def load_state(path: str) -> Dict[str, Any]:
    state = read_json_file(path, default_state())
    if not isinstance(state, dict):
        return default_state()
    state.setdefault("version", STATE_VERSION)
    state.setdefault("idle_observations", {})
    state.setdefault("managed_vms", {})
    state.setdefault("attempts", {})
    state.setdefault("cooldowns", {})
    state.setdefault("lifecycle_cache", {})
    return state


def save_state(path: str, state: Dict[str, Any]) -> None:
    state["version"] = STATE_VERSION
    atomic_write_json(path, state)


def normalize_tpu_type(raw: Any) -> str:
    value = str(raw).strip().lower()
    if value.startswith("v6-"):
        value = "v6e-" + value.split("-", 1)[1]
    elif value.startswith("v5-"):
        value = "v5p-" + value.split("-", 1)[1]
    match = TYPE_RE.match(value)
    if not match:
        raise ConfigError(f"invalid TPU type {raw!r}; expected examples: v6e-32, v6-32, v5p-64")
    family = match.group(1)
    if family not in VERSION_BY_FAMILY:
        raise ConfigError(f"unsupported TPU family in {raw!r}; manager supports v5p-* and v6e-*")
    return value


def extract_tpu_type(name: str) -> Optional[str]:
    match = NAME_TYPE_RE.search(name or "")
    if not match:
        return None
    try:
        return normalize_tpu_type(match.group(1))
    except ConfigError:
        return match.group(1).lower()


def parse_family_size(tpu_type: str) -> Tuple[str, int]:
    norm = normalize_tpu_type(tpu_type)
    family, size_s = norm.rsplit("-", 1)
    return family, int(size_s)


def zone_to_region(zone: str) -> str:
    parts = zone.split("-")
    if len(parts) < 3:
        return zone
    return "-".join(parts[:-1])


def expand_regions(regions: Sequence[str]) -> Tuple[str, ...]:
    zones: List[str] = []
    for raw in regions:
        item = str(raw).strip()
        if not item:
            continue
        if item in KNOWN_REGION_ZONES:
            zones.extend(KNOWN_REGION_ZONES[item])
        elif ZONE_RE.match(item):
            zones.append(item)
        else:
            raise ConfigError(f"unknown region/zone {item!r}")
    deduped = []
    for zone in zones:
        if zone not in deduped:
            deduped.append(zone)
    return tuple(deduped)


def is_zone_compatible(tpu_type: str, zone: str) -> bool:
    family, _ = parse_family_size(tpu_type)
    if family == "v5p":
        return zone.endswith("-a")
    if family == "v6e":
        return zone.endswith("-b")
    return False


def allowed_zones_for_type(tpu_type: str, regions: Sequence[str]) -> Tuple[str, ...]:
    scope_zones = expand_regions(regions)
    return tuple(zone for zone in scope_zones if is_zone_compatible(tpu_type, zone))


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def int_value(value: Any, default: int, minimum: Optional[int] = None) -> int:
    if value is None:
        result = default
    else:
        result = int(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"integer value {result} is below minimum {minimum}")
    return result


def load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ConfigError(f"PyYAML is not available: {YAML_IMPORT_ERROR}")
    if not os.path.exists(path):
        raise ConfigError(f"config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def load_config(path: str, dry_run_override: Optional[bool] = None) -> ManagerConfig:
    raw = load_yaml_config(path)
    regions = tuple(raw.get("regions") or DEFAULT_REGIONS)
    expand_regions(regions)  # validate once.

    demand_map = raw.get("demands") or {}
    if not isinstance(demand_map, dict):
        raise ConfigError("demands must be a mapping from TPU type to settings")
    demands: Dict[str, Demand] = {}
    for raw_type, spec in demand_map.items():
        tpu_type = normalize_tpu_type(raw_type)
        if spec is None:
            spec = {}
        if isinstance(spec, int):
            spec = {"target_idle": spec}
        if not isinstance(spec, dict):
            raise ConfigError(f"demand for {raw_type!r} must be a mapping or integer target")
        target_idle = int_value(spec.get("target_idle"), 0, minimum=0)
        max_inflight = int_value(spec.get("max_inflight"), 1, minimum=0)
        demand_regions = tuple(spec.get("regions") or regions)
        zones = allowed_zones_for_type(tpu_type, demand_regions)
        if target_idle > 0 and not zones:
            raise ConfigError(f"demand {raw_type!r} has no compatible zones in regions={list(demand_regions)}")
        demands[tpu_type] = Demand(
            tpu_type=tpu_type,
            target_idle=target_idle,
            max_inflight=max_inflight,
            zones=zones,
        )

    create_raw = raw.get("create") or {}
    reclaim_raw = raw.get("reclaim") or {}
    audit_raw = raw.get("audit") or {}
    loop_raw = raw.get("loop") or {}

    cooldown_defaults = {
        "success": 30,
        "failure": 180,
        "capacity": 900,
        "quota": 900,
        "rate_limit": 600,
        "timeout": 900,
    }
    cooldown_raw = create_raw.get("cooldown_seconds") or {}
    cooldown_seconds = dict(cooldown_defaults)
    for key, value in cooldown_raw.items():
        cooldown_seconds[str(key)] = int_value(value, cooldown_defaults.get(str(key), 180), minimum=0)

    labels_raw = create_raw.get("labels") or {"env": "prod"}
    if not isinstance(labels_raw, dict):
        raise ConfigError("create.labels must be a mapping")
    labels = {str(k): str(v) for k, v in labels_raw.items() if str(k) and str(v)}

    reclaim = {
        "allow_delete_others": bool_value(reclaim_raw.get("allow_delete_others"), False),
        "require_preemptible_or_spot": bool_value(reclaim_raw.get("require_preemptible_or_spot"), True),
        "lifecycle_cache_ttl_seconds": int_value(
            reclaim_raw.get("lifecycle_cache_ttl_seconds"),
            3600,
            minimum=0,
        ),
        "delete_non_demand": bool_value(reclaim_raw.get("delete_non_demand"), True),
        "delete_surplus_demand": bool_value(reclaim_raw.get("delete_surplus_demand"), True),
        "keep_surplus": int_value(reclaim_raw.get("keep_surplus"), 0, minimum=0),
        "idle_ttl_seconds": int_value(reclaim_raw.get("idle_ttl_minutes"), 120, minimum=0) * 60,
        "max_delete_per_loop": int_value(reclaim_raw.get("max_delete_per_loop"), 8, minimum=0),
        "delete_workers": int_value(reclaim_raw.get("delete_workers"), 4, minimum=1),
        "delete_timeout_seconds": int_value(reclaim_raw.get("delete_timeout_seconds"), 300, minimum=1),
        "confirm_idle_before_delete": bool_value(reclaim_raw.get("confirm_idle_before_delete"), True),
        "protected_name_substrings": list(reclaim_raw.get("protected_name_substrings") or []),
    }

    dry_run = bool_value(raw.get("dry_run"), False)
    if dry_run_override is not None:
        dry_run = dry_run_override

    return ManagerConfig(
        path=path,
        enabled=bool_value(raw.get("enabled"), True),
        dry_run=dry_run,
        owner=str(raw.get("owner") or "sqa"),
        base_prefix=str(raw.get("base_prefix") or "sqa-rm"),
        regions=regions,
        demands=demands,
        loop_interval_seconds=int_value(loop_raw.get("interval_seconds"), 60, minimum=1),
        max_cache_age_seconds=int_value(audit_raw.get("max_cache_age_seconds"), 300, minimum=1),
        max_create_per_loop=int_value(create_raw.get("max_create_per_loop"), 8, minimum=0),
        create_timeout_seconds=int_value(create_raw.get("create_timeout_seconds"), 900, minimum=1),
        create_workers=int_value(create_raw.get("workers"), 4, minimum=1),
        min_attempt_interval_seconds=int_value(create_raw.get("min_attempt_interval_seconds"), 60, minimum=0),
        new_vm_grace_seconds=int_value(create_raw.get("new_vm_grace_seconds"), 900, minimum=0),
        append_vm_txt=bool_value(create_raw.get("append_vm_txt"), True),
        labels=labels,
        cooldown_seconds=cooldown_seconds,
        max_cooldown_seconds=int_value(create_raw.get("max_cooldown_seconds"), 3600, minimum=0),
        reclaim=reclaim,
    )


def read_audit_cache(path: str = AUDIT_CACHE) -> Tuple[float, List[Dict[str, Any]]]:
    data = read_json_file(path, None)
    if not isinstance(data, dict):
        raise AuditCacheError(f"audit cache is invalid or missing: {path}")
    try:
        ts = float(data["ts"])
        records = list(data["records"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditCacheError(f"audit cache is missing ts/records: {path}") from exc
    return ts, records


def run_fresh_audit(timeout_seconds: int = 600) -> bool:
    logging.info("Refreshing tou audit cache via wrap_master.py --cache false")
    try:
        result = subprocess.run(
            [sys.executable, WRAP_MASTER, "--cache", "false"],
            cwd=TPU_DLS_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        logging.warning("fresh tou audit timed out after %ss", timeout_seconds)
        return False
    if result.returncode != 0:
        logging.warning("fresh tou audit failed rc=%s stderr_tail=%s", result.returncode, tail_text(result.stderr))
        return False
    return True


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def tail_text(text: Any, max_chars: int = 1000) -> str:
    text = as_text(text)
    text = text or ""
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def filtered_records(records: Iterable[Dict[str, Any]], scope_zones: Iterable[str]) -> List[Dict[str, Any]]:
    scope = set(scope_zones)
    output = []
    for record in records:
        name = str(record.get("name") or "")
        zone = str(record.get("zone") or "")
        status = str(record.get("status") or "")
        tpu_type = extract_tpu_type(name)
        if not name or not zone or not tpu_type:
            continue
        if zone not in scope:
            continue
        normalized = dict(record)
        normalized["name"] = name
        normalized["zone"] = zone
        normalized["status"] = status
        normalized["tpu_type"] = tpu_type
        output.append(normalized)
    return output


def update_idle_observations(
    state: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    cache_ts: float,
    retention_seconds: int = 7 * 24 * 3600,
) -> None:
    observations = state.setdefault("idle_observations", {})
    managed = state.setdefault("managed_vms", {})
    seen = set()
    for record in records:
        name = record["name"]
        seen.add(name)
        status = record["status"]
        entry = observations.get(name) or {}
        old_status = entry.get("last_status")
        if status == "IDLE":
            if old_status != "IDLE" or not entry.get("idle_since"):
                entry["idle_since"] = cache_ts
            entry["last_idle_seen"] = cache_ts
        else:
            entry["idle_since"] = None
        entry.update(
            {
                "name": name,
                "zone": record["zone"],
                "tpu_type": record["tpu_type"],
                "last_status": status,
                "last_status_seen": cache_ts,
                "managed": name in managed,
            }
        )
        observations[name] = entry

    cutoff = cache_ts - retention_seconds
    for name in list(observations.keys()):
        if name in seen:
            continue
        last_seen = observations[name].get("last_status_seen") or 0
        if last_seen < cutoff:
            del observations[name]


def inventory_counts(
    records: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
    config: ManagerConfig,
    now: float,
) -> Dict[str, Any]:
    by_type_status: Dict[str, Dict[str, int]] = {}
    idle_by_type_zone: Dict[str, Dict[str, int]] = {}
    seen_names = {r["name"] for r in records}

    for record in records:
        tpu_type = record["tpu_type"]
        status = record["status"]
        by_type_status.setdefault(tpu_type, {})
        by_type_status[tpu_type][status] = by_type_status[tpu_type].get(status, 0) + 1
        if status == "IDLE":
            idle_by_type_zone.setdefault(tpu_type, {})
            idle_by_type_zone[tpu_type][record["zone"]] = idle_by_type_zone[tpu_type].get(record["zone"], 0) + 1

    pending_by_type_zone: Dict[str, Dict[str, int]] = {}
    for name, entry in state.get("managed_vms", {}).items():
        if name in seen_names:
            continue
        created_at = float(entry.get("created_at") or 0)
        if created_at <= 0 or now - created_at > config.new_vm_grace_seconds:
            continue
        tpu_type = entry.get("tpu_type")
        zone = entry.get("zone")
        if not tpu_type or not zone:
            continue
        pending_by_type_zone.setdefault(tpu_type, {})
        pending_by_type_zone[tpu_type][zone] = pending_by_type_zone[tpu_type].get(zone, 0) + 1

    return {
        "by_type_status": by_type_status,
        "idle_by_type_zone": idle_by_type_zone,
        "pending_by_type_zone": pending_by_type_zone,
    }


def sum_type_zone(mapping: Dict[str, Dict[str, int]], tpu_type: str, zones: Sequence[str]) -> int:
    by_zone = mapping.get(tpu_type) or {}
    return sum(int(by_zone.get(zone, 0)) for zone in zones)


def cooldown_key(tpu_type: str, zone: str) -> str:
    return f"{tpu_type}|{zone}"


def cooldown_active(state: Dict[str, Any], tpu_type: str, zone: str, now: float) -> Optional[Dict[str, Any]]:
    entry = state.get("cooldowns", {}).get(cooldown_key(tpu_type, zone))
    if not entry:
        return None
    until = float(entry.get("until") or 0)
    if until > now:
        return entry
    return None


def attempt_too_recent(state: Dict[str, Any], tpu_type: str, zone: str, now: float, min_interval: int) -> bool:
    if min_interval <= 0:
        return False
    entry = state.get("attempts", {}).get(cooldown_key(tpu_type, zone))
    if not entry:
        return False
    last_attempt = float(entry.get("last_attempt") or 0)
    return now - last_attempt < min_interval


def choose_create_zone(
    demand: Demand,
    counts: Dict[str, Any],
    state: Dict[str, Any],
    planned_by_zone: Dict[str, int],
    config: ManagerConfig,
    now: float,
) -> Optional[str]:
    candidates = []
    idle_by_zone = counts["idle_by_type_zone"].get(demand.tpu_type) or {}
    pending_by_zone = counts["pending_by_type_zone"].get(demand.tpu_type) or {}
    for zone in demand.zones:
        if cooldown_active(state, demand.tpu_type, zone, now):
            continue
        if attempt_too_recent(state, demand.tpu_type, zone, now, config.min_attempt_interval_seconds):
            continue
        current = idle_by_zone.get(zone, 0) + pending_by_zone.get(zone, 0) + planned_by_zone.get(zone, 0)
        attempt = state.get("attempts", {}).get(cooldown_key(demand.tpu_type, zone), {})
        failures = int(attempt.get("consecutive_failures") or 0)
        last_attempt = float(attempt.get("last_attempt") or 0)
        candidates.append((current, failures, last_attempt, random.random(), zone))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][-1]


def protected_name(name: str, protected_substrings: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(str(item).lower() in lowered for item in protected_substrings)


def describe_tpu_lifecycle(name: str, zone: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        "--format=json(schedulingConfig,state)",
    ]
    checked_at = utc_now()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "checked_at": checked_at,
            "error": "timeout",
            "reclaimable_lifecycle": False,
        }
    if proc.returncode != 0:
        return {
            "ok": False,
            "checked_at": checked_at,
            "error": tail_text(proc.stderr or proc.stdout),
            "reclaimable_lifecycle": False,
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "checked_at": checked_at,
            "error": f"json_decode:{exc}",
            "reclaimable_lifecycle": False,
        }
    scheduling = payload.get("schedulingConfig") or {}
    spot = bool_value(scheduling.get("spot"), False)
    preemptible = bool_value(scheduling.get("preemptible"), False)
    return {
        "ok": True,
        "checked_at": checked_at,
        "state": payload.get("state"),
        "spot": spot,
        "preemptible": preemptible,
        "reclaimable_lifecycle": bool(spot or preemptible),
    }


def get_tpu_lifecycle(
    record: Dict[str, Any],
    state: Dict[str, Any],
    config: ManagerConfig,
    now: float,
    force: bool = False,
) -> Dict[str, Any]:
    cache = state.setdefault("lifecycle_cache", {})
    name = record["name"]
    ttl = int(config.reclaim.get("lifecycle_cache_ttl_seconds") or 0)
    cached = cache.get(name)
    if (
        not force
        and cached
        and cached.get("zone") == record["zone"]
        and ttl > 0
        and now - float(cached.get("checked_at") or 0) <= ttl
    ):
        return dict(cached)

    info = describe_tpu_lifecycle(name, record["zone"])
    info.update(
        {
            "name": name,
            "zone": record["zone"],
            "tpu_type": record.get("tpu_type"),
        }
    )
    cache[name] = info
    return info


def plan_actions(
    config: ManagerConfig,
    state: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    cache_ts: float,
    now: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    counts = inventory_counts(records, state, config, now)
    demanded_types = set(config.demands.keys())

    create_actions: List[Dict[str, Any]] = []
    planned_by_zone: Dict[str, int] = {}
    create_budget = config.max_create_per_loop
    demand_status: Dict[str, Dict[str, Any]] = {}

    for tpu_type, demand in sorted(config.demands.items()):
        current_idle = sum_type_zone(counts["idle_by_type_zone"], tpu_type, demand.zones)
        pending_new = sum_type_zone(counts["pending_by_type_zone"], tpu_type, demand.zones)
        effective_idle = current_idle + pending_new
        deficit = max(0, demand.target_idle - effective_idle)
        demand_status[tpu_type] = {
            "target_idle": demand.target_idle,
            "idle": current_idle,
            "pending_new": pending_new,
            "effective_idle": effective_idle,
            "deficit": deficit,
            "zones": list(demand.zones),
        }
        if create_budget <= 0 or deficit <= 0 or demand.max_inflight <= 0:
            continue
        attempts = min(deficit, demand.max_inflight, create_budget)
        for _ in range(attempts):
            zone = choose_create_zone(demand, counts, state, planned_by_zone, config, now)
            if zone is None:
                break
            planned_by_zone[zone] = planned_by_zone.get(zone, 0) + 1
            create_budget -= 1
            create_actions.append({"kind": "create", "tpu_type": tpu_type, "zone": zone})
            if create_budget <= 0:
                break

    observations = state.get("idle_observations", {})
    delete_actions: List[Dict[str, Any]] = []
    reclaim = config.reclaim
    if reclaim["max_delete_per_loop"] > 0:
        idle_counts_for_delete = {
            t: sum(status_counts.get("IDLE", 0) for status_counts in [counts["by_type_status"].get(t, {})])
            for t in counts["by_type_status"]
        }
        candidates = []
        for record in records:
            if record["status"] != "IDLE":
                continue
            name = record["name"]
            tpu_type = record["tpu_type"]
            if protected_name(name, reclaim["protected_name_substrings"]):
                continue
            if reclaim["require_preemptible_or_spot"]:
                lifecycle = get_tpu_lifecycle(record, state, config, now)
                if not lifecycle.get("reclaimable_lifecycle"):
                    logging.info(
                        "skip delete candidate name=%s zone=%s type=%s: not spot/preemptible or lifecycle unknown (%s)",
                        name,
                        record["zone"],
                        tpu_type,
                        lifecycle.get("error") or lifecycle.get("state") or "non_reclaimable",
                    )
                    continue
            obs = observations.get(name) or {}
            idle_since = obs.get("idle_since")
            if not idle_since:
                continue
            idle_seconds = cache_ts - float(idle_since)
            if idle_seconds < reclaim["idle_ttl_seconds"]:
                continue
            managed = name in state.get("managed_vms", {})
            if not reclaim["allow_delete_others"] and not managed:
                continue

            reason = None
            priority = 0
            if tpu_type not in demanded_types:
                if reclaim["delete_non_demand"]:
                    reason = "non_demand_long_idle"
                    priority = 0
            elif reclaim["delete_surplus_demand"]:
                demand = config.demands[tpu_type]
                surplus_limit = demand.target_idle + reclaim["keep_surplus"]
                current_idle = idle_counts_for_delete.get(tpu_type, 0)
                if current_idle > surplus_limit:
                    reason = "surplus_demand_long_idle"
                    priority = 1
            if reason is None:
                continue
            candidates.append((priority, -idle_seconds, name, record, idle_seconds, reason, managed))

        candidates.sort()
        for _priority, _neg_idle, _name, record, idle_seconds, reason, managed in candidates:
            tpu_type = record["tpu_type"]
            if reason == "surplus_demand_long_idle":
                demand = config.demands[tpu_type]
                surplus_limit = demand.target_idle + reclaim["keep_surplus"]
                current_idle = idle_counts_for_delete.get(tpu_type, 0)
                if current_idle <= surplus_limit:
                    continue
                idle_counts_for_delete[tpu_type] = current_idle - 1
            delete_actions.append(
                {
                    "kind": "delete",
                    "name": record["name"],
                    "zone": record["zone"],
                    "tpu_type": tpu_type,
                    "idle_seconds": idle_seconds,
                    "reason": reason,
                    "managed": managed,
                }
            )
            if len(delete_actions) >= reclaim["max_delete_per_loop"]:
                break

    plan_summary = {
        "counts": counts,
        "demands": demand_status,
        "create_count": len(create_actions),
        "delete_count": len(delete_actions),
    }
    return create_actions, delete_actions, plan_summary


def random_suffix(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_vm_name(tpu_type: str, base_prefix: str) -> str:
    safe_prefix = safe_base_prefix(base_prefix)
    day_tag = dt.datetime.utcnow().strftime("%m%d%H%M")
    name = f"kmh-tpuvm-{tpu_type}-spot-{safe_prefix}-{day_tag}-{random_suffix(6)}"
    if len(name) > 63:
        # TPU names are GCP resource names; keep a hard margin for long prefixes.
        keep = max(1, 63 - len(f"kmh-tpuvm-{tpu_type}-spot--{day_tag}-xxxxxx"))
        safe_prefix = safe_prefix[:keep].strip("-") or "rm"
        name = f"kmh-tpuvm-{tpu_type}-spot-{safe_prefix}-{day_tag}-{random_suffix(6)}"
    return name


def safe_base_prefix(base_prefix: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", base_prefix.lower()).strip("-") or "sqa-rm"


def is_manager_named_vm(name: str, config: ManagerConfig) -> bool:
    safe_prefix = re.escape(safe_base_prefix(config.base_prefix))
    return re.match(
        rf"^kmh-tpuvm-v[0-9]+[a-z]?-[0-9]+-spot-{safe_prefix}-[0-9]{{8}}-[a-z0-9]{{6}}$",
        name,
    ) is not None


def service_account_for_zone(zone: str) -> Optional[str]:
    return REGION_SA_MAP.get(zone_to_region(zone))


def labels_arg(labels: Dict[str, str]) -> Optional[str]:
    if not labels:
        return None
    pairs = []
    for key, value in labels.items():
        clean_key = re.sub(r"[^a-z0-9_-]+", "-", key.lower()).strip("-_")
        clean_value = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
        if clean_key and clean_value:
            pairs.append(f"{clean_key}={clean_value}")
    if not pairs:
        return None
    return "--labels=" + ",".join(pairs)


def create_tpu(action: Dict[str, Any], config: ManagerConfig, dry_run: bool) -> Dict[str, Any]:
    tpu_type = action["tpu_type"]
    zone = action["zone"]
    family, _ = parse_family_size(tpu_type)
    version = VERSION_BY_FAMILY[family]
    name = build_vm_name(tpu_type, config.base_prefix)
    service_account = service_account_for_zone(zone)
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "create",
        name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        f"--accelerator-type={tpu_type}",
        f"--version={version}",
        "--spot",
        "--quiet",
    ]
    if service_account:
        cmd.append(f"--service-account={service_account}")
    label = labels_arg(config.labels)
    if label:
        cmd.append(label)

    result = {
        "kind": "create",
        "name": name,
        "zone": zone,
        "tpu_type": tpu_type,
        "cmd": cmd,
        "dry_run": dry_run,
        "ok": False,
    }
    if dry_run:
        result["ok"] = True
        result["dry_run_only"] = True
        return result

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.create_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = as_text(exc.stdout) + "\n" + as_text(exc.stderr)
        result.update(
            {
                "ok": False,
                "error_class": "timeout",
                "returncode": None,
                "output_tail": tail_text(output),
            }
        )
        return result

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    result["returncode"] = proc.returncode
    result["output_tail"] = tail_text(output)
    if proc.returncode == 0:
        result["ok"] = True
        return result
    result["error_class"] = classify_create_error(output)
    return result


def classify_create_error(output: str) -> str:
    text = (output or "").lower()
    if "rate_limit_exceeded" in text or "rate limit" in text:
        return "rate_limit"
    if "quota" in text or "limit exceeded" in text:
        return "quota"
    if (
        "capacity" in text
        or "resource exhausted" in text
        or "resources are not available" in text
        or "zone_resource_pool_exhausted" in text
        or "availability" in text
    ):
        return "capacity"
    if "already exists" in text:
        return "failure"
    return "failure"


def set_create_result_state(state: Dict[str, Any], result: Dict[str, Any], config: ManagerConfig, now: float) -> None:
    key = cooldown_key(result["tpu_type"], result["zone"])
    attempts = state.setdefault("attempts", {})
    attempt = attempts.get(key) or {}
    attempt["last_attempt"] = now
    attempt["last_result"] = "success" if result.get("ok") else "failure"
    attempt["last_name"] = result.get("name")

    cooldowns = state.setdefault("cooldowns", {})
    if result.get("ok"):
        attempt["consecutive_failures"] = 0
        cooldown = config.cooldown_seconds.get("success", 0)
        if cooldown:
            cooldowns[key] = {
                "until": now + cooldown,
                "reason": "success",
                "last_update": now,
            }
        if not result.get("dry_run_only"):
            state.setdefault("managed_vms", {})[result["name"]] = {
                "name": result["name"],
                "zone": result["zone"],
                "tpu_type": result["tpu_type"],
                "created_at": now,
                "created_by": "tpu_request_manager",
                "owner": config.owner,
            }
        attempts[key] = attempt
        return

    failures = int(attempt.get("consecutive_failures") or 0) + 1
    attempt["consecutive_failures"] = failures
    error_class = result.get("error_class") or "failure"
    base_cooldown = config.cooldown_seconds.get(error_class, config.cooldown_seconds.get("failure", 180))
    cooldown = min(config.max_cooldown_seconds, base_cooldown * min(failures, 4))
    cooldowns[key] = {
        "until": now + cooldown,
        "reason": error_class,
        "consecutive_failures": failures,
        "last_update": now,
        "last_output_tail": result.get("output_tail", ""),
    }
    attempts[key] = attempt


def vm_record_exists(name: str) -> bool:
    try:
        with open(VM_TXT, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(f"{name}|"):
                    return True
    except OSError:
        return False
    return False


def append_vm_record_parts(name: str, zone: str, tpu_type: str) -> bool:
    if vm_record_exists(name):
        return False
    line = f"{name}|{zone}|{tpu_type}\n"
    with open(VM_TXT, "a", encoding="utf-8") as file:
        file.write(line)
    return True


def append_vm_record(result: Dict[str, Any]) -> None:
    append_vm_record_parts(result["name"], result["zone"], result["tpu_type"])


def reconcile_manager_named_vms(
    state: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    config: ManagerConfig,
    now: float,
) -> None:
    managed = state.setdefault("managed_vms", {})
    reconciled = 0
    appended = 0
    for record in records:
        name = record["name"]
        if not is_manager_named_vm(name, config):
            continue
        if name not in managed:
            managed[name] = {
                "name": name,
                "zone": record["zone"],
                "tpu_type": record["tpu_type"],
                "created_at": now,
                "created_by": "tpu_request_manager",
                "owner": config.owner,
                "reconciled_at": now,
            }
            reconciled += 1
        if config.append_vm_txt and append_vm_record_parts(name, record["zone"], record["tpu_type"]):
            appended += 1
    if reconciled or appended:
        logging.info("reconciled manager-named VMs: state=%s vm_txt=%s", reconciled, appended)


def parse_lock_time(text: str) -> Optional[float]:
    try:
        parsed = dt.datetime.strptime(text, LOCK_TIME_FMT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.timezone.utc).timestamp()


def reserved_user_from_lock(name: str, now: float, ttl_seconds: int = 30 * 60) -> Optional[str]:
    try:
        files = os.listdir(TPU_LOCK_DIR)
    except OSError:
        return None
    for filename in files:
        parts = filename.split("_")
        if len(parts) < 4:
            continue
        user = parts[0]
        vm_name = "_".join(parts[1:-2])
        time_text = f"{parts[-2]}_{parts[-1]}"
        if vm_name != name:
            continue
        lock_ts = parse_lock_time(time_text)
        if lock_ts is None:
            continue
        if 0 <= now - lock_ts <= ttl_seconds:
            return user
    return None


def confirm_remote_idle(name: str, zone: str, timeout_seconds: int = 60) -> Tuple[bool, str]:
    describe_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        "--format=value(state)",
    ]
    try:
        desc = subprocess.run(
            describe_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, "describe_timeout"
    if desc.returncode != 0:
        return False, "describe_failed"
    state = (desc.stdout or "").strip().upper()
    if state != "READY":
        return False, f"state_{state or 'unknown'}"

    remote_cmd = (
        "PID=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | head -n 1); "
        'if [ -z "$PID" ]; then echo "CHECK_RES:IDLE"; '
        'else TPU_USER=$(ps -o user= -p "$PID"); echo "CHECK_RES:BUSY|USER:$TPU_USER"; fi'
    )
    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        name,
        "--zone",
        zone,
        f"--project={PROJECT}",
        "--worker=all",
        "--ssh-flag=-n",
        "--command",
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            ssh_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, "ssh_timeout"
    if proc.returncode != 0:
        return False, "ssh_failed"
    checks = []
    for line in (proc.stdout or "").splitlines():
        if "CHECK_RES:" not in line:
            continue
        checks.append(line.split("CHECK_RES:", 1)[1].strip())
    if not checks:
        return False, "no_check_output"
    if any(item.startswith("BUSY") for item in checks):
        return False, "remote_busy"
    return True, "idle"


def delete_tpu(action: Dict[str, Any], config: ManagerConfig, dry_run: bool) -> Dict[str, Any]:
    name = action["name"]
    zone = action["zone"]
    result = {
        "kind": "delete",
        "name": name,
        "zone": zone,
        "tpu_type": action["tpu_type"],
        "reason": action["reason"],
        "idle_seconds": action["idle_seconds"],
        "managed": action["managed"],
        "dry_run": dry_run,
        "ok": False,
    }
    if dry_run:
        result["ok"] = True
        result["dry_run_only"] = True
        return result

    now = utc_now()
    reserved_by = reserved_user_from_lock(name, now)
    if reserved_by:
        result["skipped"] = True
        result["skip_reason"] = f"reserved_by_{reserved_by}"
        return result

    if config.reclaim["require_preemptible_or_spot"]:
        lifecycle = describe_tpu_lifecycle(name, zone)
        result["lifecycle"] = lifecycle
        if not lifecycle.get("reclaimable_lifecycle"):
            result["skipped"] = True
            result["skip_reason"] = "non_preemptible_non_spot"
            return result

    if config.reclaim["confirm_idle_before_delete"]:
        confirmed, reason = confirm_remote_idle(name, zone)
        if not confirmed:
            result["skipped"] = True
            result["skip_reason"] = f"not_confirmed_idle:{reason}"
            return result

    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        "--quiet",
    ]
    result["cmd"] = cmd
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.reclaim["delete_timeout_seconds"],
        )
    except subprocess.TimeoutExpired:
        result["error_class"] = "timeout"
        return result
    result["returncode"] = proc.returncode
    result["output_tail"] = tail_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    result["ok"] = proc.returncode == 0
    return result


def run_deletes(
    delete_actions: Sequence[Dict[str, Any]],
    config: ManagerConfig,
    dry_run: bool,
    events_path: str,
) -> List[Dict[str, Any]]:
    if not delete_actions:
        return []
    logging.info("Planned deletes: %s", len(delete_actions))
    for action in delete_actions:
        logging.info(
            "delete candidate name=%s zone=%s type=%s idle_minutes=%.1f reason=%s managed=%s",
            action["name"],
            action["zone"],
            action["tpu_type"],
            action["idle_seconds"] / 60,
            action["reason"],
            action["managed"],
        )
    if dry_run:
        results = [delete_tpu(action, config, dry_run=True) for action in delete_actions]
    else:
        workers = min(config.reclaim["delete_workers"], len(delete_actions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(delete_tpu, action, config, False) for action in delete_actions]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for result in results:
        append_jsonl(events_path, result)
        if result.get("ok"):
            logging.info("delete ok name=%s dry_run=%s", result["name"], result.get("dry_run"))
        elif result.get("skipped"):
            logging.info("delete skipped name=%s reason=%s", result["name"], result.get("skip_reason"))
        else:
            logging.warning("delete failed name=%s output=%s", result["name"], result.get("output_tail", ""))
    return results


def run_creates(
    create_actions: Sequence[Dict[str, Any]],
    config: ManagerConfig,
    dry_run: bool,
    state: Dict[str, Any],
    events_path: str,
) -> List[Dict[str, Any]]:
    if not create_actions:
        return []
    logging.info("Planned creates: %s", len(create_actions))
    for action in create_actions:
        logging.info("create candidate type=%s zone=%s", action["tpu_type"], action["zone"])
    if dry_run:
        results = [create_tpu(action, config, dry_run=True) for action in create_actions]
    else:
        workers = min(config.create_workers, len(create_actions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_action = {
                pool.submit(create_tpu, action, config, False): action
                for action in create_actions
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_action):
                action = future_to_action[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "kind": "create",
                            "name": None,
                            "zone": action["zone"],
                            "tpu_type": action["tpu_type"],
                            "dry_run": dry_run,
                            "ok": False,
                            "error_class": "worker_exception",
                            "output_tail": f"{type(exc).__name__}: {exc}",
                        }
                    )

    now = utc_now()
    for result in results:
        if not result.get("dry_run_only"):
            set_create_result_state(state, result, config, now)
        append_jsonl(events_path, result)
        if result.get("ok"):
            logging.info(
                "create ok name=%s zone=%s type=%s dry_run=%s",
                result["name"],
                result["zone"],
                result["tpu_type"],
                result.get("dry_run"),
            )
            if config.append_vm_txt and not result.get("dry_run_only"):
                append_vm_record(result)
        else:
            logging.warning(
                "create failed type=%s zone=%s class=%s output=%s",
                result["tpu_type"],
                result["zone"],
                result.get("error_class"),
                result.get("output_tail", ""),
            )
    return results


def prune_deleted_managed_vms(state: Dict[str, Any], records: Sequence[Dict[str, Any]], now: float) -> None:
    # Keep missing managed VMs around for the create grace period; after that the
    # audit cache is authoritative enough for planning.
    seen = {record["name"] for record in records}
    managed = state.setdefault("managed_vms", {})
    for name in list(managed.keys()):
        if name in seen:
            continue
        created_at = float(managed[name].get("created_at") or 0)
        deleted_at = managed[name].get("deleted_at")
        if deleted_at and now - float(deleted_at) > 24 * 3600:
            del managed[name]
        elif created_at and now - created_at > 24 * 3600:
            # It either disappeared or never made it into cache; stop counting it
            # as a manager-owned VM for future planning.
            managed[name]["stale_missing_since"] = managed[name].get("stale_missing_since") or now


def apply_delete_results_to_state(state: Dict[str, Any], results: Sequence[Dict[str, Any]], now: float) -> None:
    managed = state.setdefault("managed_vms", {})
    observations = state.setdefault("idle_observations", {})
    for result in results:
        if not result.get("ok") or result.get("dry_run_only"):
            continue
        name = result["name"]
        if name in managed:
            managed[name]["deleted_at"] = now
            managed[name]["delete_reason"] = result.get("reason")
        if name in observations:
            observations[name]["last_status"] = "DELETED_BY_MANAGER"
            observations[name]["idle_since"] = None
            observations[name]["deleted_at"] = now


def log_plan_summary(plan_summary: Dict[str, Any]) -> None:
    for tpu_type, item in sorted(plan_summary.get("demands", {}).items()):
        logging.info(
            "demand type=%s target_idle=%s idle=%s pending_new=%s effective_idle=%s deficit=%s zones=%s",
            tpu_type,
            item["target_idle"],
            item["idle"],
            item["pending_new"],
            item["effective_idle"],
            item["deficit"],
            ",".join(item["zones"]),
        )
    logging.info(
        "plan summary: creates=%s deletes=%s",
        plan_summary.get("create_count", 0),
        plan_summary.get("delete_count", 0),
    )


def manager_once(
    config_path: str,
    state_path: str,
    events_path: str,
    dry_run_override: Optional[bool] = None,
    refresh_if_stale: bool = False,
) -> int:
    config = load_config(config_path, dry_run_override=dry_run_override)
    dry_run = config.dry_run
    state = load_state(state_path)
    now = utc_now()

    try:
        cache_ts, raw_records = read_audit_cache(AUDIT_CACHE)
    except AuditCacheError as exc:
        logging.error("%s", exc)
        return 2

    cache_age = now - cache_ts
    if cache_age > config.max_cache_age_seconds and refresh_if_stale:
        if run_fresh_audit():
            cache_ts, raw_records = read_audit_cache(AUDIT_CACHE)
            now = utc_now()
            cache_age = now - cache_ts

    if cache_age > config.max_cache_age_seconds:
        logging.warning(
            "audit cache is stale age=%.0fs max=%ss; skipping create/delete",
            cache_age,
            config.max_cache_age_seconds,
        )
        state["last_loop"] = {
            "ts": now,
            "ts_iso": iso_ts(now),
            "status": "skipped_stale_cache",
            "cache_age_seconds": cache_age,
        }
        save_state(state_path, state)
        return 1

    scope_zones = expand_regions(config.regions)
    records = filtered_records(raw_records, scope_zones)
    update_idle_observations(state, records, cache_ts)
    prune_deleted_managed_vms(state, records, now)
    reconcile_manager_named_vms(state, records, config, now)

    if not config.enabled:
        logging.info("manager disabled in config; updating idle observations only")
        state["last_loop"] = {
            "ts": now,
            "ts_iso": iso_ts(now),
            "status": "disabled",
            "cache_age_seconds": cache_age,
        }
        save_state(state_path, state)
        return 0

    logging.info(
        "manager loop start dry_run=%s cache_age=%.0fs records_in_scope=%s",
        dry_run,
        cache_age,
        len(records),
    )
    create_actions, delete_actions, plan_summary = plan_actions(config, state, records, cache_ts, now)
    log_plan_summary(plan_summary)

    delete_results = run_deletes(delete_actions, config, dry_run, events_path)
    apply_delete_results_to_state(state, delete_results, utc_now())
    create_results = run_creates(create_actions, config, dry_run, state, events_path)

    state["last_loop"] = {
        "ts": utc_now(),
        "ts_iso": iso_ts(),
        "status": "ok",
        "dry_run": dry_run,
        "cache_age_seconds": cache_age,
        "records_in_scope": len(records),
        "planned_creates": len(create_actions),
        "planned_deletes": len(delete_actions),
        "create_ok": sum(1 for item in create_results if item.get("ok")),
        "delete_ok": sum(1 for item in delete_results if item.get("ok")),
    }
    save_state(state_path, state)
    return 0


class ExclusiveLock:
    def __init__(self, path: str):
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self) -> "ExclusiveLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another request manager is already running; lock={self.path}") from exc
        os.ftruncate(self.fd, 0)
        os.write(self.fd, f"{os.getpid()} {iso_ts()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def print_status(config_path: str, state_path: str, dry_run_override: Optional[bool]) -> int:
    config = load_config(config_path, dry_run_override=dry_run_override)
    state = load_state(state_path)
    now = utc_now()
    try:
        cache_ts, raw_records = read_audit_cache(AUDIT_CACHE)
    except AuditCacheError as exc:
        print(f"cache: ERROR {exc}")
        return 2
    cache_age = now - cache_ts
    records = filtered_records(raw_records, expand_regions(config.regions))
    update_idle_observations(state, records, cache_ts)
    create_actions, delete_actions, plan_summary = plan_actions(config, state, records, cache_ts, now)

    print(f"config: {config.path}")
    print(f"enabled={config.enabled} dry_run={config.dry_run} owner={config.owner}")
    print(f"cache_ts={iso_ts(cache_ts)} cache_age_seconds={cache_age:.0f} records_in_scope={len(records)}")
    print()
    print("Demands:")
    for tpu_type, item in sorted(plan_summary["demands"].items()):
        print(
            f"  {tpu_type}: target={item['target_idle']} idle={item['idle']} "
            f"pending_new={item['pending_new']} effective={item['effective_idle']} "
            f"deficit={item['deficit']} zones={','.join(item['zones'])}"
        )
    if not plan_summary["demands"]:
        print("  (none)")
    print()
    print(f"Planned creates: {len(create_actions)}")
    for action in create_actions[:20]:
        print(f"  create {action['tpu_type']} in {action['zone']}")
    if len(create_actions) > 20:
        print(f"  ... {len(create_actions) - 20} more")
    print()
    print(f"Planned deletes: {len(delete_actions)}")
    for action in delete_actions[:20]:
        print(
            f"  delete {action['name']} {action['zone']} {action['tpu_type']} "
            f"idle_minutes={action['idle_seconds'] / 60:.1f} reason={action['reason']}"
        )
    if len(delete_actions) > 20:
        print(f"  ... {len(delete_actions) - 20} more")
    last_loop = state.get("last_loop")
    if last_loop:
        print()
        print(f"last_loop: {last_loop}")
    return 0


def validate_config(config_path: str, dry_run_override: Optional[bool]) -> int:
    config = load_config(config_path, dry_run_override=dry_run_override)
    print(f"OK: {config.path}")
    print(f"enabled={config.enabled} dry_run={config.dry_run}")
    for tpu_type, demand in sorted(config.demands.items()):
        print(
            f"demand {tpu_type}: target_idle={demand.target_idle} "
            f"max_inflight={demand.max_inflight} zones={','.join(demand.zones)}"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Centralized TPU VM request/reclaim manager.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to request_demand.yaml")
    parser.add_argument("--state", default=DEFAULT_STATE, help="Path to request_state.json")
    parser.add_argument("--events", default=DEFAULT_EVENTS, help="Path to events.jsonl")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to request_manager.log")
    parser.add_argument("--lock", default=DEFAULT_LOCK, help="Path to lock file")
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument("--dry-run", action="store_true", help="Force dry-run regardless of config")
    dry_group.add_argument("--execute", action="store_true", help="Force real create/delete regardless of config")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="cmd", required=True)
    def add_subcommand_dry_flags(subparser: argparse.ArgumentParser) -> None:
        group = subparser.add_mutually_exclusive_group()
        group.add_argument("--dry-run", dest="cmd_dry_run", action="store_true", default=False)
        group.add_argument("--execute", dest="cmd_execute", action="store_true", default=False)

    once = sub.add_parser("once", help="Run one manager iteration")
    add_subcommand_dry_flags(once)
    once.add_argument("--refresh-if-stale", action="store_true", help="Run a fresh tou audit if the cache is stale")

    loop = sub.add_parser("loop", help="Run continuously and reload config each iteration")
    add_subcommand_dry_flags(loop)
    loop.add_argument("--refresh-if-stale", action="store_true", help="Run a fresh tou audit if the cache is stale")

    status = sub.add_parser("status", help="Print current plan without saving state or executing")
    add_subcommand_dry_flags(status)
    validate = sub.add_parser("validate-config", help="Validate config and print normalized demand")
    add_subcommand_dry_flags(validate)
    return parser


def dry_run_override_from_args(args: argparse.Namespace) -> Optional[bool]:
    parent = True if args.dry_run else False if args.execute else None
    child = None
    if getattr(args, "cmd_dry_run", False):
        child = True
    elif getattr(args, "cmd_execute", False):
        child = False
    if parent is not None and child is not None and parent != child:
        raise ConfigError("conflicting dry-run/execute flags")
    return child if child is not None else parent


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dry_override = dry_run_override_from_args(args)

    if args.cmd == "status":
        return print_status(args.config, args.state, dry_override)
    if args.cmd == "validate-config":
        return validate_config(args.config, dry_override)

    setup_logging(args.log, verbose=args.verbose)
    if args.cmd == "once":
        with ExclusiveLock(args.lock):
            return manager_once(
                args.config,
                args.state,
                args.events,
                dry_run_override=dry_override,
                refresh_if_stale=args.refresh_if_stale,
            )

    if args.cmd == "loop":
        with ExclusiveLock(args.lock):
            logging.info("request manager loop started")
            while True:
                loop_start = utc_now()
                try:
                    rc = manager_once(
                        args.config,
                        args.state,
                        args.events,
                        dry_run_override=dry_override,
                        refresh_if_stale=args.refresh_if_stale,
                    )
                except ConfigError as exc:
                    logging.error("config error: %s", exc)
                    rc = 2
                    sleep_seconds = 60
                except Exception:
                    logging.exception("manager loop crashed during iteration")
                    rc = 2
                    sleep_seconds = 60
                else:
                    try:
                        config = load_config(args.config, dry_run_override=dry_override)
                        sleep_seconds = config.loop_interval_seconds
                    except Exception:
                        sleep_seconds = 60
                elapsed = utc_now() - loop_start
                logging.info("iteration rc=%s elapsed=%.1fs sleeping=%ss", rc, elapsed, sleep_seconds)
                time.sleep(max(1, sleep_seconds))

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2)
