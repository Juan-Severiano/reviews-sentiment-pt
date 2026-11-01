# reviews-sentiment-pt

Classificador de sentimento (positivo ou negativo) para avaliações de e-commerce em português. Compara um baseline clássico (TF-IDF + Regressão Logística) com o BERTimbau ajustado, com a divisão dos dados **por produto**, para o número do teste ser confiável.

## Pergunta

Quanto um modelo de linguagem ajustado ganha sobre um baseline clássico em avaliações em português, e a que custo de latência?

## Resultados

Conjunto de teste: 11.367 avaliações de 4.265 produtos que não aparecem no treino nem na validação.

| Modelo | Acurácia | F1 macro | Latência por avaliação |
|---|---|---|---|
| Classe majoritária | 0,7083 | 0,4146 | |
| TF-IDF + Regressão Logística (`C=8`) | **0,9515** | **0,9423** | 0,16 ms (CPU, Apple Silicon) |
| BERTimbau ajustado | a medir | a medir | a medir |

O BERTimbau ainda não tem resultado aqui: o script `src/finetune.py` está pronto e foi validado em um teste curto, e o treino completo está rodando. Os números entram em um commit seguinte, junto com o modelo no Hugging Face.

O F1 macro é a métrica principal porque as classes são desbalanceadas (70,6% positivas no treino).

| Classe | Precisão | Revocação | F1 | Suporte |
|---|---|---|---|---|
| negativo | 0,8942 | 0,9457 | 0,9192 | 3.316 |
| positivo | 0,9771 | 0,9539 | 0,9654 | 8.051 |

![Matriz de confusão do baseline](results/baseline_confusion.png)

## O que garante que o número é confiável

- **Divisão por produto.** 34.115 produtos no treino, 4.265 na validação e 4.265 no teste, com **0 produtos compartilhados**. Dividir por avaliação deixaria o mesmo produto nos dois lados e poderia inflar o resultado.
- **Só o texto da avaliação.** O título ("Ótimo", "Péssimo") entrega o rótulo e ficou de fora.
- **Rótulo pela nota.** 4 ou 5 é positivo, 1 ou 2 é negativo, a nota 3 é descartada (112.008 avaliações depois disso).
- **O teste não entra na escolha.** O hiperparâmetro `C` é escolhido só pela validação. A grade foi estendida (`1` a `32`) depois que o melhor valor apareceu na borda dela.
- **Repetição medida.** 1,28% dos textos do teste aparecem idênticos no treino (avaliações curtas, como "Ótimo"). O número está em `results/dataset_stats.json`.

## Onde o baseline erra

Em `results/baseline_errors.csv` há 30 erros amostrados do teste. Os padrões que aparecem:

1. **Nota alta com texto de reclamação de entrega.** O cliente dá nota 5 ao produto e escreve que recebeu só um dos dois itens, ou que a entrega atrasou. O rótulo mede o produto, o texto mede a logística.
2. **Perguntas postadas como avaliação**, com nota baixa e sem opinião no texto.
3. **Ironia e humor**, que um modelo baseado em palavras não pega.
4. **Avaliações mistas**, com elogio e crítica na mesma frase.

Boa parte desses erros é ruído do rótulo, não falha do modelo, e isso limita o quanto qualquer modelo pode melhorar.

## Como reproduzir

Pré-requisito: [uv](https://docs.astral.sh/uv/).

```bash
# Baseline: ~10 s em CPU
uv run --python 3.12 python src/baseline.py

# Fine-tuning do BERTimbau (testado em Apple Silicon com MPS; GPU CUDA também funciona)
uv run --python 3.12 --extra train python src/finetune.py --max-train 20000 --epochs 2 --batch-size 16
```

O primeiro uso baixa o dataset e, no fine-tuning, o modelo base (~420 MB). O dataset original usa um script de carregamento que a versão atual do `datasets` recusa, então o código lê a versão em Parquet gerada pelo Hugging Face (`revision="refs/convert/parquet"`).

## Estrutura

```
src/
├── data.py          carrega o dataset, monta o rótulo e divide por produto
├── evaluation.py    métricas e matriz de confusão (usadas pelos dois modelos)
├── baseline.py      TF-IDF + Regressão Logística
└── finetune.py      fine-tuning do BERTimbau com o Trainer do Hugging Face
results/             métricas, estatísticas dos dados, gráfico e erros amostrados
```

## Limitações

- O rótulo vem da **nota**, não de anotação humana do texto.
- São avaliações de e-commerce da B2W. Não foi avaliado em outros domínios, como entregas, restaurantes ou redes sociais.
- Só distingue positivo de negativo. Avaliações neutras ou mistas ficam fora do treino.

## Licenças

Código sob [MIT](LICENSE). Dados: [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01), avaliações de produtos de e-commerce da B2W, sob CC BY 4.0. Modelo base: [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased), MIT.
