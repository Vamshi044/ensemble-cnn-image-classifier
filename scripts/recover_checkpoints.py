"""Find Stage 3B checkpoints that survived a lost runtime and move them somewhere safe.

    python scripts/recover_checkpoints.py --dry-run
    python scripts/recover_checkpoints.py --destination DIR_ON_GOOGLE_DRIVE

A hosted runtime keeps its disk only while it lives. When one is reclaimed
mid-run, whatever ``src.training.fit`` had written is still on that disk if the
machine comes back, and is gone if it does not. This script looks in every
plausible location, reads each candidate with the project's own loader, reports
exactly which architecture and epoch it represents, and copies the best copy of
each into persistent storage.

It never deletes anything, never overwrites a destination file that is at a
later epoch than the candidate, and trains nothing. Checkpoints are read with
``weights_only=True``, so nothing inside a recovered file is executed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import load_checkpoint  # noqa: E402
from src.utils import write_json  # noqa: E402

ARCHITECTURES = ("resnet18", "googlenet", "vgg11")
KINDS = ("best", "last")

# Ordered by how likely each is to hold the real thing: the working directory
# of a Colab clone first, then the places a backup would have been put by hand.
DEFAULT_SEARCH_ROOTS = (
    "checkpoints",
    "/content/ensemble-cnn-image-classifier/checkpoints",
    "/content/checkpoints",
    "/content/drive/MyDrive",
    "/content/drive/Shareddrives",
)


def candidate_names() -> tuple[str, ...]:
    return tuple(f"{a}_{k}.pt" for a in ARCHITECTURES for k in KINDS)


def scan(roots: list[Path]) -> list[Path]:
    """Every file under ``roots`` whose name is a Stage 3B checkpoint name.

    Directories that do not exist are skipped rather than raised on: most of
    the default roots are absent on any given machine, and that is the normal
    case rather than an error.
    """
    wanted = set(candidate_names())
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.pt"):
            if path.name not in wanted:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def describe(path: Path) -> dict:
    """Read a candidate and say what it is.

    A corrupt or truncated file is reported rather than raised on, because one
    unreadable file must not hide the others from the scan.
    """
    record: dict = {"path": str(path), "bytes": path.stat().st_size,
                    "readable": False}
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - see docstring
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    stored = checkpoint.get("config") or {}
    record.update(
        readable=True,
        architecture=checkpoint.get("architecture"),
        epoch=int(checkpoint.get("epoch", -1)) + 1,
        best_metric=checkpoint.get("best_metric"),
        best_epoch=checkpoint.get("best_epoch"),
        has_optimizer="optimizer_state" in checkpoint,
        has_scheduler="scheduler_state" in checkpoint,
        has_scaler="scaler_state" in checkpoint,
        has_rng="rng_state" in checkpoint,
        configured_epochs=stored.get("epochs"),
        resumable=all(k in checkpoint for k in ("optimizer_state", "scheduler_state")),
    )
    return record


def best_per_slot(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Pick one file per (architecture, kind): whichever reached the later epoch."""
    chosen: dict[tuple[str, str], dict] = {}
    for record in records:
        if not record["readable"]:
            continue
        stem = Path(record["path"]).name.removesuffix(".pt")
        architecture, _, kind = stem.rpartition("_")
        if architecture != record["architecture"]:
            record["warning"] = (
                f"filename says {architecture!r} but the checkpoint stores "
                f"{record['architecture']!r}; skipped"
            )
            continue
        incumbent = chosen.get((architecture, kind))
        if incumbent is None or record["epoch"] > incumbent["epoch"]:
            chosen[(architecture, kind)] = record
    return chosen


def recover(chosen: dict[tuple[str, str], dict], destination: Path,
            dry_run: bool) -> list[dict]:
    """Copy each chosen file into ``destination``, never regressing what is there.

    A destination already holding a later epoch is left alone: recovery must
    not be able to turn a more complete run into a less complete one.
    """
    actions: list[dict] = []
    for (architecture, kind), record in sorted(chosen.items()):
        source = Path(record["path"])
        target = destination / f"{architecture}_{kind}.pt"
        action = {"architecture": architecture, "kind": kind,
                  "source": str(source), "target": str(target),
                  "epoch": record["epoch"]}

        if target.is_file() and source.resolve() == target.resolve():
            action["result"] = "already in place"
        elif target.is_file():
            existing = describe(target)
            if existing["readable"] and existing["epoch"] >= record["epoch"]:
                action["result"] = (
                    f"kept existing copy at epoch {existing['epoch']} "
                    f"(candidate is epoch {record['epoch']})"
                )
            else:
                action["result"] = "replace an earlier copy"
        else:
            action["result"] = "copy"

        if action["result"] in ("copy", "replace an earlier copy"):
            if dry_run:
                action["result"] = "would " + action["result"]
            else:
                destination.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".tmp")
                shutil.copy2(source, temp)
                temp.replace(target)
                action["result"] = "copied" if action["result"] == "copy" else (
                    "replaced an earlier copy")
        actions.append(action)
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", nargs="+", type=Path, default=None,
                        help="directories to search recursively")
    parser.add_argument("--destination", type=Path, default=None,
                        help="persistent directory to copy recovered checkpoints into")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what was found and copy nothing")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "results" / "checkpoint_recovery.json")
    args = parser.parse_args(argv)

    roots = [Path(r) for r in (args.search or DEFAULT_SEARCH_ROOTS)]
    print("searching:")
    for root in roots:
        print(f"  {'[  ok  ]' if root.is_dir() else '[absent]'} {root}")

    records = [describe(path) for path in scan(roots)]

    print(f"\n{len(records)} candidate file(s):")
    for record in records:
        if not record["readable"]:
            print(f"  UNREADABLE {record['path']} - {record['error']}")
            continue
        print(f"  {record['architecture']:<10} epoch {record['epoch']:>3}"
              f"  best {record['best_metric']}"
              f"  resumable={record['resumable']}"
              f"  {record['bytes'] / 1024**2:.0f} MB  {record['path']}")

    chosen = best_per_slot(records)
    missing = [f"{a}_{k}" for a in ARCHITECTURES for k in KINDS
               if (a, k) not in chosen]

    actions: list[dict] = []
    if chosen and args.destination is not None:
        actions = recover(chosen, args.destination, args.dry_run)
        print("\nrecovery:")
        for action in actions:
            print(f"  {action['architecture']}_{action['kind']:<5} "
                  f"epoch {action['epoch']:>3}  {action['result']}")
    elif chosen:
        print("\nno --destination given, so nothing was copied.")

    if missing:
        print(f"\nnot found: {', '.join(missing)}")

    report = write_json(
        {
            "search_roots": [str(r) for r in roots],
            "candidates": records,
            "selected": {f"{a}_{k}": r["path"] for (a, k), r in chosen.items()},
            "missing": missing,
            "actions": actions,
            "destination": None if args.destination is None else str(args.destination),
            "dry_run": args.dry_run,
        },
        args.report,
    )
    print(f"\nreport {report}")

    if not chosen:
        print("\nNo Stage 3B checkpoint was found. Nothing was recovered.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
