import argparse
from ablation_runner_utils import Variant, add_common_args, run_variants


def main():
    parser = argparse.ArgumentParser(description="Run anchor-mode ablations.")
    add_common_args(parser)
    args = parser.parse_args()

    variants = [
        Variant("no_anchor", ("--coeff_prior_anch", "0.0", "--anchor_mode", "fixed", "--fixed_alpha", "0.0")),
        Variant("true_only", ("--anchor_mode", "fixed", "--fixed_alpha", "0.0")),
        Variant("pred_only", ("--anchor_mode", "fixed", "--fixed_alpha", "1.0")),
        Variant("fixed_half", ("--anchor_mode", "fixed", "--fixed_alpha", "0.5")),
        Variant("dynamic_metric", ("--anchor_mode", "dynamic", "--schedule_type", "metric")),
    ]
    run_variants(args, variants, suite_name="anchor_modes")


if __name__ == "__main__":
    main()
