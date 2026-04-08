import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "Evaluations"


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation pipelines end-to-end")
    parser.add_argument("--skip_alpaca", action="store_true")
    parser.add_argument("--skip_beavertails", action="store_true")
    parser.add_argument("--skip_truthfulqa", action="store_true")
    parser.add_argument("--truthfulqa_sample_size", type=int, default=30)
    parser.add_argument("--beavertails_sample_size", type=int, default=10)
    args = parser.parse_args()

    if not args.skip_alpaca:
        run([sys.executable, str(EVAL_DIR / "run_alpaca_eval_limited.py")])

    if not args.skip_beavertails:
        run(
            [
                sys.executable,
                str(EVAL_DIR / "evaluate_beavertails_safety_score.py"),
                "--csv_path",
                str(REPO_ROOT / "Pak3H" / "PakBeavertails.csv"),
                "--sample_size",
                str(args.beavertails_sample_size),
            ]
        )

    if not args.skip_truthfulqa:
        run(
            [
                sys.executable,
                str(EVAL_DIR / "TruthfulQA" / "gptj_end_to_end.py"),
                "--input_path",
                str(REPO_ROOT / "Pak3H" / "PakTruthfulqa.csv"),
                "--output_path",
                str(EVAL_DIR / "TruthfulQA" / "outputs" / "gptj_answers.csv"),
                "--summary_path",
                str(EVAL_DIR / "TruthfulQA" / "outputs" / "gptj_summary.csv"),
                "--sample_size",
                str(args.truthfulqa_sample_size),
            ]
        )


if __name__ == "__main__":
    main()
