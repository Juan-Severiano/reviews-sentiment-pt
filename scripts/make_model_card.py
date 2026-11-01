"""Gera o model card (README.md do repositório do modelo no Hugging Face) a partir dos resultados.

Uso: uv run python scripts/make_model_card.py --repo-id usuario/bertimbau-b2w-sentiment
Lê results/baseline.json, results/bertimbau.json e results/dataset_stats.json e escreve
outputs/bertimbau/final/README.md. Nada é digitado à mão: todo número vem dos arquivos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%".replace(".", ",")


def n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="usuario/nome do modelo no Hub")
    parser.add_argument("--code-url", default="", help="URL do repositório de código no GitHub")
    args = parser.parse_args()

    base = json.loads((RESULTS / "baseline.json").read_text())
    tuned = json.loads((RESULTS / "bertimbau.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    if not tuned["full_test_set"]:
        raise SystemExit("O model card só deve ser gerado com o teste completo (--max-test 0).")

    bt, bb = tuned["test"], base["test"]
    cmp = tuned["comparison_with_baseline"]
    split = stats["splits"]
    code_line = f"- Código e reprodução: {args.code_url}\n" if args.code_url else ""

    card = f"""---
language: pt
license: mit
base_model: neuralmind/bert-base-portuguese-cased
datasets:
- ruanchaves/b2w-reviews01
metrics:
- f1
- accuracy
pipeline_tag: text-classification
tags:
- sentiment-analysis
- portuguese
- e-commerce
---

# BERTimbau para sentimento de avaliações em português

[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) ajustado para
classificar avaliações de e-commerce como **positivas** ou **negativas**.

## Resultado

Conjunto de teste de {n(split["test"]["rows"])} avaliações de produtos que **não aparecem no treino
nem na validação** (divisão por produto).

| Modelo | F1 macro | Acurácia | Latência por avaliação |
|---|---|---|---|
| Classe majoritária | {base["majority_class_baseline_test"]["f1_macro"]:.4f} | {base["majority_class_baseline_test"]["accuracy"]:.4f} | |
| TF-IDF + Regressão Logística (baseline) | {bb["f1_macro"]:.4f} | {bb["accuracy"]:.4f} | {base["latency_ms_per_review_single"]} ms (CPU) |
| **BERTimbau ajustado (este modelo)** | **{bt["f1_macro"]:.4f}** | **{bt["accuracy"]:.4f}** | {tuned["latency_ms_per_review_single"]} ms ({tuned["device"]}) |

Diferença contra o baseline: **{cmp["delta_f1_macro_pp"]:+.2f} p.p. de F1 macro** e
**{cmp["delta_accuracy_pp"]:+.2f} p.p. de acurácia**.

| Classe | Precisão | Revocação | F1 | Suporte |
|---|---|---|---|---|
"""
    for name, m in bt["per_class"].items():
        card += f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"

    card += f"""
## Como usar

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{args.repo_id}")
classifier("Chegou antes do prazo e o produto é excelente!")
```

## Dados

[B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) (CC BY 4.0).
Rótulo derivado da nota: 4 ou 5 = positivo, 1 ou 2 = negativo, nota 3 descartada.
Entrada: apenas o texto da avaliação (o título entrega o rótulo e foi excluído).
Divisão 80/10/10 **por produto** (semente {stats["seed"]}), com {stats["products_shared_between_splits"]}
produtos compartilhados entre conjuntos. {pct(stats["test_texts_also_in_train_share"])} dos textos
do teste aparecem idênticos no treino (avaliações curtas como "Ótimo").
Proporção no treino: {pct(split["train"]["class_share"]["positivo"])} positivas e
{pct(split["train"]["class_share"]["negativo"])} negativas.

## Treinamento

- Amostra estratificada de {n(tuned["rows"]["train"])} avaliações de treino e {n(tuned["rows"]["val"])} de validação.
- {tuned["args"]["epochs"]:g} épocas, taxa de aprendizado {tuned["args"]["lr"]:g}, lote {tuned["args"]["batch_size"]}, `max_len` {tuned["args"]["max_len"]}.
- Melhor época escolhida por F1 macro na validação (F1 macro de validação: {tuned["val_f1_macro"]}).
- Hardware: {tuned["device"]}; tempo de treino {tuned["training_time_s"] / 60:.1f} min.

## Limitações

- O rótulo vem da **nota**, não de anotação humana do texto. Há avaliações com nota alta e texto
  reclamando de entrega, e perguntas postadas como avaliação. Parte dos "erros" é ruído do rótulo.
- Treinado em avaliações de e-commerce da B2W (janeiro a maio de 2018). Não foi avaliado em outros
  domínios, como restaurantes, entregas ou redes sociais.
- Só distingue positivo de negativo. Avaliações neutras ou mistas não foram usadas no treino.
- **Não separa produto de entrega.** Como o rótulo é a nota do produto, uma avaliação que elogia o
  produto e reclama do atraso ("Produto ótimo, mas a entrega atrasou 10 dias") sai **positiva**.
  Para medir a entrega separadamente seria preciso um modelo por aspecto.
- Não use para decisões automáticas sobre pessoas.

{code_line}"""
    out = ROOT / "outputs" / "bertimbau" / "final" / "README.md"
    out.write_text(card, encoding="utf-8")
    print("model card escrito em", out)


if __name__ == "__main__":
    main()
