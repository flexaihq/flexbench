import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import evaluate
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from flexbench.config import DatasetConfig
from flexbench.dataset.text import TextDataset
from flexbench.utils import get_logger

log = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def normalize_text(text: str | list[str]) -> str | list[str]:
    """Normalize text by stripping and formatting into sentences."""
    if isinstance(text, list):
        return [normalize_text(t) for t in text]
    return "\n".join(nltk.sent_tokenize(text.strip()))


def compute_rouge_chunk(args):
    """Compute ROUGE scores for a chunk of predictions and references."""
    metric, preds, targets = args
    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )
    return result


def compute_rouge_scores(preds: list[str], refs: list[str]) -> tuple[dict, int]:
    """Compute ROUGE scores and token statistics."""
    metric = evaluate.load("rouge")

    log.debug(f"Computing ROUGE scores for {len(preds)} predictions and {len(refs)} references")
    if not preds or not refs:
        log.warning(f"Empty predictions ({len(preds)}) or references ({len(refs)})")
        return {}, 0

    n_workers = cpu_count()
    chunk_size = len(preds) // n_workers + 1
    chunks = [
        (metric, preds[i : i + chunk_size], refs[i : i + chunk_size])
        for i in range(0, len(preds), chunk_size)
    ]

    with Pool() as pool:
        results_list = pool.map(compute_rouge_chunk, chunks)

    if not results_list:
        log.warning("No ROUGE results computed")
        return {}, 0

    scores = {k: [] for k in results_list[0].keys()}
    for result in results_list:
        for k, v in result.items():
            scores[k].extend(v)

    return scores, sum(len(p) for p in preds)


def run_accuracy_check(
    model_path: str,
    dataset_config: DatasetConfig,
    mlperf_accuracy_file: str | Path,
    export_txt: bool = False,
    export_json: bool = False,
    dtype: str = "int32",
) -> dict:
    """Run accuracy checking and return metrics."""
    log.info(f"Starting accuracy evaluation: {mlperf_accuracy_file=}")

    if isinstance(mlperf_accuracy_file, str):
        mlperf_accuracy_file = Path(mlperf_accuracy_file)

    results_dir = mlperf_accuracy_file.parent

    accuracy_txt_path = results_dir / "accuracy.txt"
    accuracy_json_path = results_dir / "accuracy_details.json"
    accuracy_results_path = results_dir / "accuracy_results.json"

    log.info(
        f"Loading references from {dataset_config.path} ({dataset_config.split}) using columns: "
        f"input='{dataset_config.input_column}', output='{dataset_config.output_column}'"
    )

    dataset = TextDataset(dataset_config=dataset_config, model_path=model_path)
    reference_data = dataset.get_references()
    log.info(f"Loaded {len(reference_data.references)} references")
    if not reference_data.references:
        log.error("No references found in dataset. Check dataset configuration.")
        return {}

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    log.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        use_fast=True,
        add_prefix_space=None if dataset.model_type == "deepseek" else False,
    )

    gen_tok_len = 0
    tokens_list = []
    valid_indices = []
    try:
        with open(mlperf_accuracy_file) as f:
            preds_json = json.load(f)
    except FileNotFoundError:
        log.error(f"MLPerf accuracy file not found: {mlperf_accuracy_file}")
        return {"error": f"MLPerf accuracy file not found: {mlperf_accuracy_file}"}

    samples = []
    for pred in tqdm(preds_json, desc="Decoding predictions (hex -> token_id)"):
        idx = pred["qsl_idx"]
        if idx >= len(reference_data.references) or any(
            s["reference"] == reference_data.references[idx] for s in samples
        ):
            continue
        tokens = [
            t
            for t in np.frombuffer(bytes.fromhex(pred["data"]), dtype)
            if 0 <= t <= tokenizer.vocab_size
        ]
        gen_tok_len += len(tokens)
        tokens_list.append(tokens)
        valid_indices.append(idx)

    log.info(
        f"Decoding {len(tokens_list)} predictions (token_id -> text) using tokenizer... "
        "This may take a while."
    )
    predictions_decoded = tokenizer.batch_decode(tokens_list, skip_special_tokens=True)

    log.debug(f"Processing {len(predictions_decoded)} decoded predictions")
    samples = [
        {
            "system_prompt": reference_data.system_prompts[idx],
            "input": reference_data.inputs[idx],
            "reference": reference_data.references[idx],
            "prediction": normalize_text(pred_text),
        }
        for idx, pred_text in zip(valid_indices, predictions_decoded)
    ]
    log.debug(f"Created {len(samples)} sample pairs")

    log.info(f"Computing ROUGE scores for {len(samples)} prediction-reference pairs")
    predictions = [s["prediction"] for s in samples]
    targets = [normalize_text(s["reference"]) for s in samples]

    if not predictions or not targets:
        log.error("No valid prediction-reference pairs found")
        return {
            "error": "No valid samples",
            "gen_len": 0,
            "gen_num": len(samples),
            "gen_tok_len": gen_tok_len,
            "tokens_per_sample": 0,
        }

    scores, total_len = compute_rouge_scores(predictions, targets)
    metrics = {
        **{k: round(np.mean(v) * 100, 4) for k, v in scores.items()},
        "gen_len": total_len,
        "gen_num": len(samples),
        "gen_tok_len": gen_tok_len,
        "tokens_per_sample": round(gen_tok_len / len(samples), 1),
    }

    if export_txt:
        with open(accuracy_txt_path, "w") as f:
            f.write("\nResults\n\n" + str(metrics))
            log.info(f"Results exported to {accuracy_txt_path}")

    if export_json:
        pd.DataFrame(
            [
                {
                    **sample,
                    "prediction": pred,
                    **{k: round(v[i] * 100, 4) for k, v in scores.items()},
                }
                for i, (sample, pred) in enumerate(zip(samples, predictions))
            ]
        ).to_json(accuracy_json_path, indent=2, orient="records")
        log.info(f"Detailed results exported to {accuracy_json_path}")

    with open(accuracy_results_path, "w") as f:
        json.dump(metrics, f, indent=2)
        log.info(f"Accuracy results exported to {accuracy_results_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run accuracy evaluation")
    parser.add_argument("--mlperf-accuracy-file", type=str, help="Path to the MLPerf accuracy file")
    parser.add_argument("--model-path", type=str, help="Path to the model")
    parser.add_argument(
        "--json-export", action="store_true", help="Export detailed results to JSON"
    )
    parser.add_argument("--dtype", type=str, default="int32", help="Data type for token IDs")

    parser.add_argument("--dataset-path", required=True, help="Path to the dataset")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use")
    parser.add_argument("--input-column", required=True, help="Input column name")
    parser.add_argument("--output-column", required=True, help="Output/reference column name")
    parser.add_argument("--system-prompt-column", help="System prompt column name")
    args = parser.parse_args()

    dataset_config = DatasetConfig(
        path=args.dataset_path,
        split=args.dataset_split,
        input_column=args.input_column,
        output_column=args.output_column,
        system_prompt_column=args.system_prompt_column,
    )

    run_accuracy_check(
        model_path=args.model_path,
        dataset_config=dataset_config,
        mlperf_accuracy_file=args.mlperf_accuracy_file,
        export_txt=True,
        export_json=args.json_export,
        dtype=args.dtype,
    )
