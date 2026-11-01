"""Carrega as avaliações da B2W, monta o rótulo de sentimento e divide os dados por produto.

Rótulo: nota 4 ou 5 = positivo (1), nota 1 ou 2 = negativo (0). A nota 3 é descartada.
Texto: só `review_text`. O título (por exemplo "Ótimo" ou "Péssimo") entrega o rótulo.
Divisão: 80% treino, 10% validação, 10% teste, **sempre por produto**, para que nenhum
produto apareça em mais de um conjunto.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET = "ruanchaves/b2w-reviews01"
SEED = 42
LABEL_NAMES = {0: "negativo", 1: "positivo"}


def load_labeled() -> pd.DataFrame:
    from datasets import load_dataset

    # O dataset original usa um script de carregamento, que o `datasets` moderno não aceita.
    # A branch `refs/convert/parquet` é a versão em Parquet gerada pelo próprio Hugging Face.
    df = load_dataset(DATASET, revision="refs/convert/parquet", split="train").to_pandas()
    df = df[["product_id", "review_text", "overall_rating"]].copy()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0]
    df = df[df["overall_rating"] != 3]
    df["label"] = (df["overall_rating"] >= 4).astype(int)
    return df.drop_duplicates(subset=["product_id", "review_text"]).reset_index(drop=True)


def _share(df: pd.DataFrame) -> dict[str, float]:
    counts = df["label"].value_counts(normalize=True)
    return {LABEL_NAMES[k]: round(float(counts.get(k, 0.0)), 4) for k in LABEL_NAMES}


def split_by_product(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(df, groups=df["product_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=SEED)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["product_id"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

    train_texts = set(train["review_text"].str.lower())
    stats = {
        "dataset": DATASET,
        "seed": SEED,
        "rows_after_labeling": int(len(df)),
        "splits": {
            name: {
                "rows": int(len(part)),
                "products": int(part["product_id"].nunique()),
                "class_share": _share(part),
            }
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "products_shared_between_splits": int(
            len(set(train["product_id"]) & set(test["product_id"]))
            + len(set(train["product_id"]) & set(val["product_id"]))
        ),
        "test_texts_also_in_train_share": round(
            float(test["review_text"].str.lower().isin(train_texts).mean()), 4
        ),
    }
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        stats,
    )
