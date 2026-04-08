import os
import shutil
import subprocess
import sys
import time
import json
import random
from pathlib import Path
import tempfile

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS_DIR = REPO_ROOT / "Evaluations"
RESULTS_DIR = EVALUATIONS_DIR / "alpaca_eval_results"

ENGLISH_MODEL_OUTPUTS = RESULTS_DIR / "your_model_english.json"
ENGLISH_REFERENCE_OUTPUTS = RESULTS_DIR / "reference_english.json"
URDU_MODEL_OUTPUTS = RESULTS_DIR / "your_model_urdu.json"
URDU_REFERENCE_OUTPUTS = RESULTS_DIR / "reference_urdu.json"

ENGLISH_OUTPUT_DIR = RESULTS_DIR / "en_eval"
URDU_OUTPUT_DIR = RESULTS_DIR / "ur_eval"

ANNOTATOR_CONFIG = EVALUATIONS_DIR / "alpaca_eval_gpt4o_mini_limited.yaml"

FN_METRIC = "get_winrate"
SORT_BY = "win_rate"

# ==============================
# Safety settings
# ==============================
MAX_RETRIES = 10
INITIAL_BACKOFF_SECONDS = 30
SLEEP_BETWEEN_RUNS_SECONDS = 60

# ==============================
# Sample 70% of dataset
# ==============================
def sample_dataset(input_path: Path, ratio=0.7):

    with open(input_path) as f:
        data = json.load(f)

    sample_size = int(len(data) * ratio)
    sampled = random.sample(data, sample_size)

    sampled_path = input_path.parent / f"sampled_{input_path.name}"

    with open(sampled_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"Using {sample_size}/{len(data)} samples from {input_path.name}")

    return sampled_path


# ==============================
# Resolve alpaca_eval binary
# ==============================
def resolve_alpaca_eval_binary():

    candidates = [
        shutil.which("alpaca_eval"),
        str(Path.home() / "Library/Python/3.14/bin/alpaca_eval"),
        str(Path(sys.executable).resolve().parent / "alpaca_eval"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise FileNotFoundError("alpaca_eval executable not found")


def resolve_prompt_template_path() -> str:
    candidates = [
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "alpaca_eval"
        / "evaluators_configs"
        / "alpaca_eval_clf_gpt4_turbo"
        / "alpaca_eval_clf.txt",
        Path.home()
        / "Library"
        / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "lib"
        / "python"
        / "site-packages"
        / "alpaca_eval"
        / "evaluators_configs"
        / "alpaca_eval_clf_gpt4_turbo"
        / "alpaca_eval_clf.txt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "Could not locate alpaca_eval prompt template. Ensure alpaca_eval is installed."
    )


def build_runtime_annotator_config() -> str:
    with open(ANNOTATOR_CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for _, cfg in config.items():
        cfg["prompt_template"] = resolve_prompt_template_path()

    tmp_dir = Path(tempfile.mkdtemp(prefix="alpaca_eval_cfg_"))
    tmp_cfg = tmp_dir / "annotator_config.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return str(tmp_cfg)


# ==============================
# Detect rate limit errors
# ==============================
def is_rate_limit_error(output):

    indicators = [
        "rate_limit_exceeded",
        "too many requests",
        "error code: 429",
    ]

    lowered = output.lower()

    return any(i in lowered for i in indicators)


# ==============================
# Run evaluation
# ==============================
def run_with_retries(label, model_outputs, reference_outputs, output_dir):

    alpaca_eval_bin = resolve_alpaca_eval_binary()

    runtime_annotator_config = build_runtime_annotator_config()

    command = [
        alpaca_eval_bin,
        "evaluate",
        "--model_outputs",
        str(model_outputs),
        "--reference_outputs",
        str(reference_outputs),
        "--annotators_config",
        runtime_annotator_config,
        "--fn_metric",
        FN_METRIC,
        "--sort_by",
        SORT_BY,
        "--name",
        f"your_model_{label.lower()}",
        "--output_path",
        str(output_dir),
    ]

    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in your environment.")

    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):

        print("\n==============================")
        print(f"Starting {label} evaluation")
        print(f"Attempt {attempt}/{MAX_RETRIES}")
        print("==============================")

        process = subprocess.Popen(
            command,
            cwd=EVALUATIONS_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines = []

        for line in process.stdout:
            print(f"[{label}] {line}", end="")
            output_lines.append(line)

        return_code = process.wait()
        combined_output = "".join(output_lines)

        if return_code == 0:
            print(f"\n{label} evaluation finished successfully")
            return

        if is_rate_limit_error(combined_output):

            print(f"\nRate limit detected. Sleeping {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2
            continue

        raise RuntimeError(f"{label} evaluation failed")


# ==============================
# Verify files
# ==============================
def verify_inputs():

    files = [
        ENGLISH_MODEL_OUTPUTS,
        ENGLISH_REFERENCE_OUTPUTS,
        URDU_MODEL_OUTPUTS,
        URDU_REFERENCE_OUTPUTS,
    ]

    missing = [str(f) for f in files if not f.exists()]

    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))


# ==============================
# Main
# ==============================
def main():

    verify_inputs()

    sampled_en_model = sample_dataset(ENGLISH_MODEL_OUTPUTS, 0.7)
    sampled_en_ref = sample_dataset(ENGLISH_REFERENCE_OUTPUTS, 0.7)

    sampled_ur_model = sample_dataset(URDU_MODEL_OUTPUTS, 0.7)
    sampled_ur_ref = sample_dataset(URDU_REFERENCE_OUTPUTS, 0.7)

    run_with_retries(
        "English",
        sampled_en_model,
        sampled_en_ref,
        ENGLISH_OUTPUT_DIR,
    )

    print(f"\nSleeping {SLEEP_BETWEEN_RUNS_SECONDS}s before Urdu...")
    time.sleep(SLEEP_BETWEEN_RUNS_SECONDS)

    run_with_retries(
        "Urdu",
        sampled_ur_model,
        sampled_ur_ref,
        URDU_OUTPUT_DIR,
    )

    print("\nAll evaluations completed")


if __name__ == "__main__":
    main()