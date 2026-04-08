import argparse
import warnings
from pathlib import Path

import pandas as pd

from truthfulqa import metrics
from truthfulqa import models
from truthfulqa import utilities


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / 'Pak3H' / 'PakTruthfulqa.csv'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'Evaluations' / 'TruthfulQA' / 'outputs' / 'gptj_answers.csv'
DEFAULT_SUMMARY_PATH = REPO_ROOT / 'Evaluations' / 'TruthfulQA' / 'outputs' / 'gptj_summary.csv'


def resolve_column(columns, requested, candidates, field_name):
    if requested is not None:
        if requested in columns:
            return requested
        raise KeyError(
            f"Requested {field_name} column '{requested}' not found. "
            f"Available columns: {list(columns)}"
        )

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise KeyError(
        f"Could not infer {field_name} column. Tried {candidates}. "
        f"Available columns: {list(columns)}"
    )


def resolve_legacy_name(columns, requested, legacy_map):
    if requested in columns:
        return requested
    mapped = legacy_map.get(requested)
    if mapped and mapped in columns:
        warnings.warn(
            f"Column '{requested}' not found, using '{mapped}' instead.",
            stacklevel=2,
        )
        return mapped
    return requested


def format_summary(results_frame):
    # Keep only numeric columns for aggregation.
    numeric = results_frame[[
        col for col in results_frame.columns
        if (col != 'Context') and pd.api.types.is_numeric_dtype(results_frame[col])
    ]]

    cols = []
    for col in numeric.columns:
        split = col.split(' ')
        cols.append((split[0], ' '.join(split[1:])))
    numeric.columns = pd.MultiIndex.from_tuples(cols)

    summary = numeric.mean(axis=0)
    summary = summary.reset_index().rename(columns={'level_0': 'Model', 'level_1': 'Metric', 0: 'Value'})

    preferred_metrics = [
        'MC1', 'MC2', 'MC3',
        'bleu acc', 'rouge1 acc',
        'BLEURT acc',
        'TI truth acc', 'TI info acc', 'TI composite',
    ]
    filtered = summary[summary['Metric'].isin(preferred_metrics)]
    if filtered.empty:
        filtered = summary.dropna(subset=['Value'])

    return pd.pivot_table(filtered, 'Value', 'Model', 'Metric')


def main():
    parser = argparse.ArgumentParser(description='Single-file GPT-J end-to-end TruthfulQA evaluation runner')
    parser.add_argument('--input_path', type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument('--output_path', type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument('--summary_path', type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument('--cache_dir', type=str, default='./hf_cache')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--preset', type=str, default='qa')
    parser.add_argument('--metrics', nargs='+', default=['mc', 'bleu', 'rouge', 'ti'])
    parser.add_argument('--sample_size', type=int, help='Limit total rows evaluated across ALL metrics (for quick tests)')
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--language', choices=['en', 'ur'], default='en')
    parser.add_argument('--question_col', type=str)
    parser.add_argument('--best_answer_col', type=str)
    parser.add_argument('--correct_answers_col', type=str)
    parser.add_argument('--incorrect_answers_col', type=str)
    parser.add_argument('--answer_col', type=str, default='gptj')
    parser.add_argument('--use_existing_answers', action='store_true')

    # TI controls
    parser.add_argument('--ti_delay', type=float, default=0.2)
    parser.add_argument('--ti_sample_size', type=int)
    parser.add_argument('--ti_seed', type=int, default=42)
    parser.add_argument('--ti_question_col', type=str)

    args = parser.parse_args()

    torch_device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'

    questions = utilities.load_questions(filename=args.input_path)

    if args.sample_size is not None:
        n = min(int(args.sample_size), len(questions))
        questions = questions.sample(n=n, random_state=args.sample_seed).reset_index(drop=True)
        print(f'Global sample_size={n}: running all metrics on {n} rows (seed={args.sample_seed})')

    legacy_map = {
        'ctx_instruction': 'ctx_Question',
        'ctx_instruction_ur': 'ctx_Question_ur',
        'ctx_output': 'ctx_Best Answer',
        'ctx_output_ur': 'ctx_Best Answer_ur',
    }

    args.question_col = resolve_legacy_name(questions.columns, args.question_col, legacy_map)
    args.answer_col = resolve_legacy_name(questions.columns, args.answer_col, legacy_map)
    args.ti_question_col = resolve_legacy_name(questions.columns, args.ti_question_col, legacy_map)

    ur_suffix = '_ur' if args.language == 'ur' else ''
    question_col = resolve_column(
        questions.columns,
        args.question_col,
        [f'ctx_Question{ur_suffix}', 'Question', 'orig_Question'],
        'question',
    )
    best_answer_col = resolve_column(
        questions.columns,
        args.best_answer_col,
        [f'ctx_Best Answer{ur_suffix}', 'Best Answer', 'orig_Best Answer'],
        'best answer',
    )
    correct_answers_col = resolve_column(
        questions.columns,
        args.correct_answers_col,
        [f'ctx_Correct Answers{ur_suffix}', 'Correct Answers', 'orig_Correct Answers'],
        'correct answers',
    )
    incorrect_answers_col = resolve_column(
        questions.columns,
        args.incorrect_answers_col,
        [f'ctx_Incorrect Answers{ur_suffix}', 'Incorrect Answers', 'orig_Incorrect Answers'],
        'incorrect answers',
    )

    # Normalize to canonical TruthfulQA field names expected by metric functions.
    questions['Question'] = questions[question_col].fillna('').astype(str)
    questions['Best Answer'] = questions[best_answer_col].fillna('').astype(str)
    questions['Correct Answers'] = questions[correct_answers_col].fillna('').astype(str)
    questions['Incorrect Answers'] = questions[incorrect_answers_col].fillna('').astype(str)

    print('Resolved evaluation columns:')
    print(f"  language={args.language}")
    print(f"  question_col={question_col}")
    print(f"  best_answer_col={best_answer_col}")
    print(f"  correct_answers_col={correct_answers_col}")
    print(f"  incorrect_answers_col={incorrect_answers_col}")

    if args.ti_question_col is None:
        ti_question_col = 'Question'
    else:
        if args.ti_question_col not in questions.columns:
            raise KeyError(f"TI question column '{args.ti_question_col}' not found in input file")
        ti_question_col = args.ti_question_col
    print(f"  ti_question_col={ti_question_col}")

    # If evaluating user-provided answers (e.g., ctx_output / ctx_output_ur), map to model key column.
    if args.use_existing_answers:
        if args.answer_col not in questions.columns:
            raise KeyError(f"Answer column '{args.answer_col}' not found in input file")
        questions['gptj'] = questions[args.answer_col].fillna('').astype(str)

    if not args.use_existing_answers:
        print('Running gptj generation...')
        models.run_GPTJ(
            questions,
            tag='gptj',
            preset=args.preset,
            device=torch_device,
            cache_dir=args.cache_dir,
            check_path=None,
        )
        utilities.save_questions(questions, args.output_path)
    else:
        print(f"Using existing answers from column '{args.answer_col}' as gptj outputs.")
        utilities.save_questions(questions, args.output_path)

    if 'mc' in args.metrics and not args.use_existing_answers:
        print('Running MC metrics...')
        models.run_probs_GPTJ(
            questions,
            tag='gptj',
            preset=args.preset,
            device=torch_device,
            cache_dir=args.cache_dir,
            check_path=None,
        )
        utilities.save_questions(questions, args.output_path)
    elif 'mc' in args.metrics and args.use_existing_answers:
        warnings.warn("Skipping MC because --use_existing_answers was set (no generation/probability pass).", stacklevel=2)

    ran_bleu_rouge = False
    for metric_name in args.metrics:
        if metric_name == 'mc':
            continue
        if metric_name in ['bleu', 'rouge']:
            if ran_bleu_rouge:
                continue
            print('Running BLEU/ROUGE...')
            metrics.run_bleu_and_rouge('gptj', questions)
            ran_bleu_rouge = True
            utilities.save_questions(questions, args.output_path)
        elif metric_name == 'bleurt':
            print('Running BLEURT...')
            metrics.run_BLEURT('gptj', questions, cache_dir=args.cache_dir)
            utilities.save_questions(questions, args.output_path)
        elif metric_name == 'ti':
            print('Running TI (GPT-J judge)...')
            metrics.run_TI_gptj(
                'gptj',
                questions,
                device=torch_device,
                cache_dir=args.cache_dir,
                delay=args.ti_delay,
                sample_size=args.ti_sample_size,
                random_seed=args.ti_seed,
                question_col=ti_question_col,
            )
            utilities.save_questions(questions, args.output_path)
        else:
            warnings.warn(f'Metric {metric_name} is not supported in this single-file runner, skipping.', stacklevel=2)

    utilities.save_questions(questions, args.output_path)

    summary = format_summary(questions)
    summary.to_csv(args.summary_path)

    print('\n=== Summary Metrics ===')
    if summary.empty:
        print('No summary metrics available for the selected configuration.')
    else:
        print(summary)
    print(f'Saved answers to {args.output_path}')
    print(f'Saved summary to {args.summary_path}')


if __name__ == '__main__':
    main()



# pip install -r requirements.txt
# pip install -e .

# python gptj_end_to_end.py
# --input_path truthfulqa_urdu-2.csv
# --output_path ti_en_out.csv
# --summary_path ti_en_summary.csv
# --cache_dir ./hf_cache
# --device 0
# --metrics ti
# --use_existing_answers
# --language en
# --question_col ctx_Question
# --answer_col "ctx_Best Answer"
# --ti_question_col ctx_Question
# --ti_sample_size 817
# --ti_seed 42
# --ti_delay 0.2

# python gptj_end_to_end.py
# --input_path truthfulqa_urdu-2.csv
# --output_path ti_ur_out.csv
# --summary_path ti_ur_summary.csv
# --cache_dir ./hf_cache
# --device 0
# --metrics ti
# --use_existing_answers
# --language ur
# --question_col ctx_Question_ur
# --answer_col "ctx_Best Answer_ur"
# --ti_question_col ctx_Question_ur
# --ti_sample_size 817
# --ti_seed 42
# --ti_delay 0.2



# Model,TI composite,TI info acc,TI truth acc
# gptj,1.3266136220971434,0.042839657282741736,0.30966952264381886


# ===== TruthfulQA Honesty Results =====
# Samples evaluated : 817
# Truthful rate     : 30.97%
# Informativeness   : 4.28%
# Honesty (TI)      : 1.33

# === Summary Metrics ===
# Metric  TI composite  TI info acc  TI truth acc
# Model                                          
# gptj        1.326614      0.04284       0.30967
# Saved answers to ti_ur_out.csv
# Saved summary to ti_ur_summary.csv


# Model,TI composite,TI info acc,TI truth acc
# gptj,5.5106526100055575,0.2460220318237454,0.22399020807833536


# Samples evaluated : 817
# Truthful rate     : 22.40%
# Informativeness   : 24.60%
# Honesty (TI)      : 5.51

# === Summary Metrics ===
# Metric  TI composite  TI info acc  TI truth acc
# Model                                          
# gptj        5.510653     0.246022       0.22399





# python gptj_end_to_end.py \
#   --input_path truthfulqa_urdu-2.csv \
#   --output_path full_en_out.csv \
#   --summary_path full_en_summary.csv \
#   --cache_dir ./hf_cache \
#   --device 0 \
#   --metrics mc bleu rouge bleurt ti \
#   --language en \
#   --question_col ctx_Question \
#   --best_answer_col "ctx_Best Answer" \
#   --correct_answers_col "ctx_Correct Answers" \
#   --incorrect_answers_col "ctx_Incorrect Answers" \
#   --ti_question_col ctx_Question \
#   --ti_sample_size 817 \
#   --ti_seed 42 \
#   --ti_delay 0.2


# python gptj_end_to_end.py \
#   --input_path truthfulqa_urdu-2.csv \
#   --output_path full_ur_out.csv \
#   --summary_path full_ur_summary.csv \
#   --cache_dir ./hf_cache \
#   --device 0 \
#   --metrics mc bleu rouge bleurt ti \
#   --language ur \
#   --question_col ctx_Question_ur \
#   --best_answer_col "ctx_Best Answer_ur" \
#   --correct_answers_col "ctx_Correct Answers_ur" \
#   --incorrect_answers_col "ctx_Incorrect Answers_ur" \
#   --ti_question_col ctx_Question_ur \
#   --ti_sample_size 817 \
#   --ti_seed 42 \
#   --ti_delay 0.2


#ALL METRICS ON SMALL SAMPLE (5) FOR QUICK TESTING
# python gptj_end_to_end.py \
#   --input_path truthfulqa_urdu-2.csv \
#   --output_path test_ur_out.csv \
#   --summary_path test_ur_summary.csv \
#   --cache_dir ./hf_cache \
#   --device 0 \
#   --metrics mc bleu rouge ti \
#   --sample_size 200 \
#   --language ur \
#   --question_col ctx_Question_ur \
#   --best_answer_col "ctx_Best Answer_ur" \
#   --correct_answers_col "ctx_Correct Answers_ur" \
#   --incorrect_answers_col "ctx_Incorrect Answers_ur" \
#   --ti_question_col ctx_Question_ur \
#   --ti_seed 42 \
#   --ti_delay 0.2


# python gptj_end_to_end.py \
#   --input_path truthfulqa_urdu-2.csv \
#   --output_path test_en_out.csv \
#   --summary_path test_en_summary.csv \
#   --cache_dir ./hf_cache \
#   --device 0 \
#   --metrics mc bleu rouge ti \
#   --sample_size 817 \
#   --language en \
#   --question_col ctx_Question \
#   --best_answer_col "ctx_Best Answer" \
#   --correct_answers_col "ctx_Correct Answers" \
#   --incorrect_answers_col "ctx_Incorrect Answers" \
#   --ti_question_col ctx_Question \
#   --ti_seed 42 \
#   --ti_delay 0.2




