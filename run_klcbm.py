"""Launcher for KLCBM runs."""


import subprocess
import sys

from tracker_utils import parse_args, load_rows, filter_rows, clean_row, fold_ids_from_scope


SCRIPT = "./models/klcbm/train_model.py"
SCRIPT = "-m models.klcbm.train_model.py"

def build_cmd(row: dict) -> list[str]:
    row = clean_row(row)

    cmd = [
        sys.executable,
        "-m",
        "models.klcbm.train_model",
        "--dataset", str(row["dataset"]),
        "--model", str(row["backbone"]),
        "--optimizer", str(row["optimizer"]),
        "--lr", str(row["lr"]),
        "--coeff_attr", str(row["coeff_attr"]),
        "--coeff_py", str(row["coeff_py"]),
        "--save_dir", "./trained_models/"+(row['model']).lower()+"/"+(row['stage']).lower()+"/", #klcbm/coeff_search/",
        "--override_hparams"
    ]

    return cmd


def main():
    args = parse_args()
    rows = load_rows(args.tracker, args.sheet)
    rows = filter_rows(
        rows,
        model="KLCBM",
        stage=args.stage,
        dataset=args.dataset,
        backbone=args.backbone,
        run_id=args.run_id,
    )

    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        folds = fold_ids_from_scope(str(row["fold_scope"]))
        for fold in folds:
            cmd = build_cmd(row) + ["--only_fold", str(fold)]
            print("RUN:", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
