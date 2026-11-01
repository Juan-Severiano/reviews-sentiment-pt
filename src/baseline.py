"""Baseline clássico: TF-IDF (1 e 2 gramas) + Regressão Logística.

É a régua do projeto. O modelo ajustado só vale se superar este número no mesmo conjunto de teste.
Uso: uv run python src/baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data import load_labeled, split_by_product
from evaluation import metrics, plot_confusion

RESULTS = Path(__file__).resolve().parent.parent / "results"
C_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train, val, test, stats = split_by_product(load_labeled())
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("linhas:", {k: v["rows"] for k, v in stats["splits"].items()})
    print("proporção por classe (treino):", stats["splits"]["train"]["class_share"])

    majority = DummyClassifier(strategy="most_frequent").fit(train["review_text"], train["label"])
    majority_metrics = metrics(test["label"], majority.predict(test["review_text"]))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200_000
    )
    x_train = vectorizer.fit_transform(train["review_text"])
    x_val = vectorizer.transform(val["review_text"])
    x_test = vectorizer.transform(test["review_text"])

    best_c, best_f1, best_model = None, -1.0, None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", solver="liblinear")
        model.fit(x_train, train["label"])
        f1 = f1_score(val["label"], model.predict(x_val), average="macro")
        print(f"C={c}: F1 macro na validação = {f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1, best_model = c, f1, model

    test_pred = best_model.predict(x_test)
    test_metrics = metrics(test["label"], test_pred)

    sample = test["review_text"].head(300).tolist()
    start = time.perf_counter()
    for text in sample[:100]:
        best_model.predict(vectorizer.transform([text]))
    single_ms = (time.perf_counter() - start) / 100 * 1000
    start = time.perf_counter()
    best_model.predict(vectorizer.transform(sample))
    batch_per_s = len(sample) / (time.perf_counter() - start)

    result = {
        "model": (
            "TF-IDF (1-2 gramas, 200 mil atributos) + "
            "Regressão Logística (liblinear, class_weight=balanced)"
        ),
        "chosen_C": best_c,
        "val_f1_macro": round(float(best_f1), 4),
        "test": test_metrics,
        "majority_class_baseline_test": {
            "accuracy": majority_metrics["accuracy"],
            "f1_macro": majority_metrics["f1_macro"],
        },
        "latency_ms_per_review_single": round(single_ms, 3),
        "throughput_reviews_per_s_batch": round(batch_per_s, 1),
        "hardware": "CPU (Apple Silicon, 16 GB)",
    }
    (RESULTS / "baseline.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    plot_confusion(
        test_metrics["confusion_matrix"],
        RESULTS / "baseline_confusion.png",
        f"TF-IDF + LR (F1 macro {test_metrics['f1_macro']})",
    )

    wrong = test.assign(pred=test_pred)
    wrong = wrong[wrong["label"] != wrong["pred"]].sample(
        n=min(30, int((wrong["label"] != wrong["pred"]).sum())), random_state=42
    )
    wrong["review_text"] = wrong["review_text"].str.slice(0, 240)
    wrong[["overall_rating", "label", "pred", "review_text"]].to_csv(
        RESULTS / "baseline_errors.csv", index=False
    )

    print("\n== TESTE ==")
    print(
        "classe majoritária:",
        majority_metrics["accuracy"],
        "acc |",
        majority_metrics["f1_macro"],
        "F1 macro",
    )
    print(
        "TF-IDF + LR       :",
        test_metrics["accuracy"],
        "acc |",
        test_metrics["f1_macro"],
        "F1 macro",
    )
    print(f"latência: {single_ms:.2f} ms por avaliação | {batch_per_s:.0f} avaliações/s em lote")


if __name__ == "__main__":
    main()
