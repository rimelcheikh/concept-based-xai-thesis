import subprocess
import sys

from tracker_utils import parse_args, load_rows, filter_rows, clean_row, fold_ids_from_scope


SCRIPT = "./models/pacbm/train_model.py"
SCRIPT = "-m models.pacbm.train_model.py"

def build_cmd(row: dict) -> list[str]:
    row = clean_row(row)

    cmd = [
        sys.executable,
        "-m",
        "models.pacbm.train_model",
        "--dataset", str(row["dataset"]),
        "--model", str(row["backbone"]),
        "--optimizer", str(row["optimizer"]),
        "--lr", str(row["lr"]),
        "--coeff_l_a_CE", str(row["coeff_l_a_CE"]),
        "--coeff_l_cls_CE", str(row["coeff_l_cls_CE"]),
        "--coeff_prior_anch", str(row["coeff_prior_anch"]),
        "--save_dir", "./trained_models/pacbm_2/"+(row['model']).lower()+"/"+(row['stage']).lower()+"/", #pacbm/coeff_search/",
    ]

    if row["dataset"] == "AwA2":
        if "start_mse" in row:
            cmd += ["--start_mse", str(row["start_mse"])]
        if "end_mse" in row:
            cmd += ["--end_mse", str(row["end_mse"])]
    else:
        if "start_bce" in row:
            cmd += ["--start_mse", str(row["start_bce"])]
        if "end_bce" in row:
            cmd += ["--end_mse", str(row["end_bce"])]

    if "ema_momentum" in row:
        cmd += ["--ema_momentum", str(row["ema_momentum"])]

    use_ema = str(row.get("use_ema_prior", "True")).lower() == "true"
    cmd.append("--use_ema_prior" if use_ema else "--no_use_ema_prior")

    return cmd


def main():
    args = parse_args()
    rows = load_rows(args.tracker, args.sheet)
    rows = filter_rows(
        rows,
        model="PACBM",
        stage=args.stage,
        dataset=args.dataset,
        backbone=args.backbone,
        run_id=args.run_id,
    )

    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        folds = fold_ids_from_scope(str(row["fold_scope"]), True)
        for fold in folds:
            cmd = build_cmd(row) + ["--only_fold", str(fold)]
            print("RUN:", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()