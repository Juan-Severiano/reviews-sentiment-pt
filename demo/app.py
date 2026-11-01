"""Demo local, em Gradio, do classificador de sentimento.

Uso:  uv run --extra app python demo/app.py
Baixa o modelo do Hugging Face. Para usar a pasta local do treino:
      MODEL_ID=outputs/bertimbau/final uv run --extra app python demo/app.py
"""

from __future__ import annotations

import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "codevjuan/bertimbau-b2w-sentiment")
classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)

EXAMPLES = [
    "Chegou antes do prazo e o produto é excelente. Recomendo!",
    "O produto veio quebrado e a loja não respondeu nenhuma das minhas mensagens.",
    "Demorou mais do que o combinado, mas o produto é bom e funciona direitinho.",
    "Não gostei. A qualidade é muito inferior ao que aparece nas fotos.",
]


def predict(text: str) -> dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {}
    out = classifier(text[:2000])
    scores = out[0] if isinstance(out[0], list) else out
    return {item["label"]: float(item["score"]) for item in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Avaliação (em português)"),
    outputs=gr.Label(num_top_classes=2, label="Sentimento"),
    title="Sentimento de avaliações em português",
    description=(
        "BERTimbau ajustado em avaliações de e-commerce da B2W. "
        "Prevê se a avaliação é positiva ou negativa. Treinado com notas 1-2 (negativo) "
        "e 4-5 (positivo); avaliações neutras não fazem parte do treino. "
        "Atenção: o rótulo vem da nota do produto, então uma avaliação que elogia o produto "
        "e reclama da entrega tende a sair positiva."
    ),
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
