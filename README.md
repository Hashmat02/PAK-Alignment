# PAK-Alignment

Anonymous ACL submission repo for multilingual alignment and evaluation in Urdu and English.

This repository is kept intentionally small for reviewers and future users:
- [Pak3H/](Pak3H) contains the paper datasets used in the experiments.
- [Evaluations/](Evaluations) contains the evaluation entrypoints.
- [Evaluations/TruthfulQA/](Evaluations/TruthfulQA) contains the consolidated TruthfulQA runners.
- [Pak3H.pdf](Pak3H.pdf) is the paper PDF.

## What Is Included

- Dataset files needed to reproduce the paper.
- One consolidated dependency file: [requirements.txt](requirements.txt).
- Reproducible command-line evaluation scripts.

## What Is Excluded

- Demo notebooks and one-off local experiment files.
- Duplicate TruthfulQA trees.
- Generated Alpaca outputs and local caches.
- API keys, machine-specific absolute paths, and other private setup.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Set the OpenAI key before running Alpaca or TruthfulQA judge-based steps:

```bash
export OPENAI_API_KEY="your_key_here"
```

If you are running the local Llama/Qwen-style TruthfulQA runner, also set a Hugging Face token:

```bash
export HF_TOKEN="your_hf_token_here"
```

## Data

The main paper datasets are in [Pak3H/](Pak3H):
- [Pak3H/PakAlpaca.csv](Pak3H/PakAlpaca.csv)
- [Pak3H/PakBeavertails.csv](Pak3H/PakBeavertails.csv)
- [Pak3H/PakTruthfulqa.csv](Pak3H/PakTruthfulqa.csv)

The TruthfulQA runner also expects the contextual fields already present in the CSV. The code normalizes English and Urdu columns automatically.

## Reviewers

If you only want to inspect the datasets and commands, start here:
- [README.md](README.md)
- [Pak3H/](Pak3H)
- [Evaluations/run_all_evaluations.py](Evaluations/run_all_evaluations.py)

If you want to rerun the experiments exactly as packaged, use the commands below.

## Simple Commands

Run everything bundled in the repo:

```bash
python3 Evaluations/run_all_evaluations.py
```

Alpaca-style pairwise evaluation:

```bash
python3 Evaluations/run_alpaca_eval_limited.py
```

BeaverTails safety score:

```bash
python3 Evaluations/evaluate_beavertails_safety_score.py \
  --csv_path Pak3H/PakBeavertails.csv \
  --sample_size 100
```

TruthfulQA with the GPT-J judge workflow used in the paper:

```bash
python3 Evaluations/TruthfulQA/gptj_end_to_end.py \
  --input_path Pak3H/PakTruthfulqa.csv \
  --output_path Evaluations/TruthfulQA/outputs/gptj_answers.csv \
  --summary_path Evaluations/TruthfulQA/outputs/gptj_summary.csv \
  --metrics mc bleu rouge bleurt ti \
  --language en \
  --question_col ctx_Question \
  --best_answer_col "ctx_Best Answer" \
  --correct_answers_col "ctx_Correct Answers" \
  --incorrect_answers_col "ctx_Incorrect Answers" \
  --ti_question_col ctx_Question \
  --ti_sample_size 817 \
  --ti_seed 42 \
  --ti_delay 0.2
```

The same runner can be used for Urdu by switching the column names:

```bash
python3 Evaluations/TruthfulQA/gptj_end_to_end.py \
  --input_path Pak3H/PakTruthfulqa.csv \
  --output_path Evaluations/TruthfulQA/outputs/gptj_answers_ur.csv \
  --summary_path Evaluations/TruthfulQA/outputs/gptj_summary_ur.csv \
  --metrics mc bleu rouge bleurt ti \
  --language ur \
  --question_col ctx_Question_ur \
  --best_answer_col "ctx_Best Answer_ur" \
  --correct_answers_col "ctx_Correct Answers_ur" \
  --incorrect_answers_col "ctx_Incorrect Answers_ur" \
  --ti_question_col ctx_Question_ur \
  --ti_sample_size 817 \
  --ti_seed 42 \
  --ti_delay 0.2
```

## Swapping Hugging Face Models

The single-file runner is the one to use when you want to swap HF causal LMs such as Llama 3 8B Instruct or Qwen2.5 7B Instruct:

```bash
python3 Evaluations/TruthfulQA/llama_single_file_end_to_end.py \
  --input_path Pak3H/PakTruthfulqa.csv \
  --output_path Evaluations/TruthfulQA/outputs/llama3_answers.csv \
  --summary_path Evaluations/TruthfulQA/outputs/llama3_summary.csv \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --chat_template_mode auto \
  --metrics mc bleu rouge ti \
  --language en \
  --question_col ctx_Question \
  --best_answer_col "ctx_Best Answer" \
  --correct_answers_col "ctx_Correct Answers" \
  --incorrect_answers_col "ctx_Incorrect Answers" \
  --ti_question_col ctx_Question \
  --sample_size 100 \
  --ti_sample_size 100
```

For Qwen2.5 7B Instruct, only change the model name:

```bash
python3 Evaluations/TruthfulQA/llama_single_file_end_to_end.py \
  --input_path Pak3H/PakTruthfulqa.csv \
  --output_path Evaluations/TruthfulQA/outputs/qwen_answers.csv \
  --summary_path Evaluations/TruthfulQA/outputs/qwen_summary.csv \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --chat_template_mode auto \
  --metrics mc bleu rouge ti \
  --language en \
  --question_col ctx_Question \
  --best_answer_col "ctx_Best Answer" \
  --correct_answers_col "ctx_Correct Answers" \
  --incorrect_answers_col "ctx_Incorrect Answers" \
  --ti_question_col ctx_Question
```

Notes for HF model swapping:
- Use `--model_name` to pass the Hugging Face model id.
- Keep `--chat_template_mode auto` for instruction-tuned chat models.
- Set `--chat_template_mode off` if you want plain prompt formatting.
- Set `HF_TOKEN` if the model requires gated access.

## Output Policy

Generated outputs are written under ignored `outputs/` folders. If you want to run a fresh experiment, point `--output_path` and `--summary_path` at your own location.


## Citation

Citation information will be added after review.