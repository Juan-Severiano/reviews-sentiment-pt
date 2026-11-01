"""Fine-tuning do BERTimbau para classificar o sentimento de avaliações em português.

Usa a mesma divisão por produto e o mesmo conjunto de teste do baseline (src/baseline.py).

Execução completa (GPU recomendada):
    uv run --extra train python src/finetune.py --max-train 20000 --epochs 2

Teste de fumaça, rápido e com resultados fora de results/:
    uv run --extra train python src/finetune.py --max-train 400 --max-val 200 --max-test 200 \
        --epochs 1 --max-len 64 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import LABEL_NAMES, SEED, load_labeled, split_by_product
from evaluation import metrics, plot_confusion

MODEL = "neuralmind/bert-base-portuguese-cased"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"


def stratified_sample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    return (
        df.groupby("label", group_keys=False)
        .sample(frac=n / len(df), random_state=SEED)
        .reset_index(drop=True)
    )


def device_name() -> str:
    if torch.cuda.is_available():
        return f"GPU CUDA ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    return "CPU"


def to_dataset(df: pd.DataFrame, tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_pandas(df[["review_text", "label"]], preserve_index=False)
    return ds.map(
        lambda batch: tokenizer(batch["review_text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["review_text"],
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def measure_latency(model, tokenizer, texts: list[str], max_len: int) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def encode(batch: list[str]):
        return tokenizer(
            batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        model(**encode(texts[:4]))
        sync()
        single = texts[:100]
        start = time.perf_counter()
        for text in single:
            model(**encode([text]))
            sync()
        single_ms = (time.perf_counter() - start) / len(single) * 1000

        batch_texts = texts[:512]
        start = time.perf_counter()
        for i in range(0, len(batch_texts), 32):
            model(**encode(batch_texts[i : i + 32]))
        sync()
        per_s = len(batch_texts) / (time.perf_counter() - start)
    return single_ms, per_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=20000, help="0 usa todo o treino")
    parser.add_argument("--max-val", type=int, default=4000)
    parser.add_argument("--max-test", type=int, default=0, help="0 usa o teste completo")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--tag", default="bertimbau")
    args = parser.parse_args()

    set_seed(SEED)
    result_dir = RESULTS if args.tag == "bertimbau" else OUTPUTS / args.tag
    result_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, _ = split_by_product(load_labeled())
    train = stratified_sample(train, args.max_train)
    val = stratified_sample(val, args.max_val)
    full_test = not args.max_test or args.max_test >= len(test)
    test = stratified_sample(test, args.max_test)
    print(f"treino {len(train)} | validação {len(val)} | teste {len(test)} | {device_name()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds, val_ds, test_ds = (to_dataset(d, tokenizer, args.max_len) for d in (train, val, test))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={name: idx for idx, name in LABEL_NAMES.items()},
    )

    # Aquecimento de 10% dos passos, calculado à mão: `warmup_ratio` saiu do transformers 5
    # e `warmup_steps` só aceita proporção a partir dele.
    steps_per_epoch = -(-len(train) // args.batch_size)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * args.epochs))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUTS / args.tag / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            seed=SEED,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    start = time.perf_counter()
    trainer.train()
    training_s = time.perf_counter() - start
    val_f1 = trainer.evaluate()["eval_f1_macro"]

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    test_metrics = metrics(test["label"], preds)
    single_ms, per_s = measure_latency(
        trainer.model, tokenizer, test["review_text"].tolist(), args.max_len
    )

    baseline_path = RESULTS / "baseline.json"
    comparison = None
    if full_test and baseline_path.exists():
        base = json.loads(baseline_path.read_text())["test"]
        comparison = {
            "baseline_f1_macro": base["f1_macro"],
            "finetuned_f1_macro": test_metrics["f1_macro"],
            "delta_f1_macro_pp": round((test_metrics["f1_macro"] - base["f1_macro"]) * 100, 2),
            "baseline_accuracy": base["accuracy"],
            "finetuned_accuracy": test_metrics["accuracy"],
            "delta_accuracy_pp": round((test_metrics["accuracy"] - base["accuracy"]) * 100, 2),
        }

    result = {
        "model": MODEL,
        "device": device_name(),
        "args": vars(args),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "full_test_set": full_test,
        "training_time_s": round(training_s, 1),
        "val_f1_macro": round(float(val_f1), 4),
        "test": test_metrics,
        "latency_ms_per_review_single": round(single_ms, 2),
        "throughput_reviews_per_s_batch32": round(per_s, 1),
        "comparison_with_baseline": comparison,
    }
    (result_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    plot_confusion(
        test_metrics["confusion_matrix"],
        result_dir / f"{args.tag}_confusion.png",
        f"BERTimbau (F1 macro {test_metrics['f1_macro']})",
    )
    wrong = test.assign(pred=preds)
    wrong = wrong[wrong["label"] != wrong["pred"]]
    wrong = wrong.sample(n=min(30, len(wrong)), random_state=SEED)
    wrong["review_text"] = wrong["review_text"].str.slice(0, 240)
    wrong[["overall_rating", "label", "pred", "review_text"]].to_csv(
        result_dir / f"{args.tag}_errors.csv", index=False
    )

    final = OUTPUTS / args.tag / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    print("\n== TESTE ==")
    print(f"F1 macro {test_metrics['f1_macro']} | acurácia {test_metrics['accuracy']}")
    if comparison:
        print(f"contra o baseline: {comparison['delta_f1_macro_pp']:+} p.p. de F1 macro")
    print(f"treino: {training_s / 60:.1f} min | latência: {single_ms:.1f} ms por avaliação")
    print(f"modelo salvo em {final}")


if __name__ == "__main__":
    main()
