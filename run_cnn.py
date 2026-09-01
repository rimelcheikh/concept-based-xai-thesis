"""Launcher for CNN runs."""


import subprocess
import sys

from tracker_utils import parse_args, load_rows, filter_rows, clean_row, fold_ids_from_scope


def build_cmd(row: dict) -> list[str]:
    row = clean_row(row)

    cmd = [
        sys.executable,
        "-m",
        "models.cnn.train_model",  # replace with the real module path if different
        "--dataset", str(row["dataset"]),
        "--model", str(row["backbone"]),
        "--optimizer", str(row["optimizer"]),
        "--lr", str(row["lr"]),
        "--save_dir", "./trained_models/" + str(row["model"]).lower() + "/" + str(row["stage"]).lower() + "/",
        "--override_hparams",
    ]
    return cmd


def main():
    args = parse_args()
    rows = load_rows(args.tracker, args.sheet)
    rows = filter_rows(
        rows,
        model="CNN",
        stage=args.stage,
        dataset=args.dataset,
        backbone=args.backbone,
        run_id=args.run_id,
    )

    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        try:
            row = clean_row(row)
            if args.dataset == "aPY":
                folds = fold_ids_from_scope(str(row["fold_scope"]),True)
            else:
                folds = fold_ids_from_scope(str(row["fold_scope"]))
                
            for fold in folds:
                cmd = build_cmd(row) + ["--only_fold", str(fold)]
                print("RUN:", " ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
        except Exception as exc:
            print(f"SKIP malformed row: run_id={row.get('run_id', '<missing>')} error={exc}")


if __name__ == "__main__":
    main()