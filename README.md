# TPU Request Manager

This manager keeps GCP TPU VM inventory close to `request_demand.yaml`.
It only creates/deletes TPU VM names. It does not create or manage xibo aliases.

Typical commands:

```bash
cd /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_request_manager
python3 request_manager.py validate-config
python3 request_manager.py status
python3 request_manager.py once --dry-run
```

To run continuously in tmux:

```bash
cd /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_request_manager
bash run_request_manager_tmux.sh
```

`request_demand.yaml` is reloaded every loop. Keep `dry_run: true` while
checking the plan; set `dry_run: false` when the plan is correct.

The manager reads inventory from:

```text
/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/.tpu_audit_records.json
```

That cache is produced by `wrap_master.py` / `yizhitou`. If it is stale, the
manager skips create/delete unless started with `--refresh-if-stale`.

## unified_infra integration

Every time the manager **successfully creates a TPU VM**, it wakes the
`unified_infra` daemon to schedule a job onto that card:

```text
infra signal <vm_name> --zone <zone> --type <tpu_type>
```

- Best-effort: a signaling failure is logged but never fails/blocks creation
  (signals also queue until the daemon runs, and the daemon's 5-min idle sweep
  is a backstop for any lost signal).
- Concurrency-safe: multiple request-manager instances may run at once; the
  `infra signal` command appends to the daemon inbox under an exclusive lock, and
  the daemon ignores a signal for a card that already has an active job, so two
  jobs never land on one card.
- Controls: `INFRA_SIGNAL=0` disables signaling; `INFRA_BIN=<path>` overrides the
  infra launcher (default `…/work/unified_infra/bin/infra`).

Reclaim protection:

- A TPU is eligible for deletion only after continuous IDLE observations meet
  `reclaim.idle_ttl_minutes`.
- When `reclaim.require_preemptible_or_spot` is true, the manager also checks
  `gcloud tpu-vm describe` and protects every non-preemptible, non-spot TPU.
