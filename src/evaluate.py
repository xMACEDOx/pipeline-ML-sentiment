"""
Etapa 5 (Avaliacao): le data/gold/test.parquet e o modelo salvo em
models/sentiment_pipeline.joblib, calcula as metricas finais e salva em
models/evaluation_metrics.json + uma imagem da matriz de confusao
(models/confusion_matrix.png).

Por que esse numero e diferente (e mais confiavel) do que ja vimos:
as metricas do train.py (validacao cruzada) e do compare_models.py (holdout)
foram sempre calculadas usando so o train.parquet - servem pra escolher e
ajustar o modelo, nao pra dizer como ele se comporta de verdade. O
test.parquet nunca foi tocado ate agora (nem no treino, nem na validacao
cruzada, nem na comparacao de modelos) - entao essa e a UNICA metrica que
reflete honestamente o que aconteceria em producao, com dado que o modelo
nunca viu de nenhuma forma.

Uso:
    python src/evaluate.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "gold" / "test.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

LABELS = ["negative", "positive"]


def main() -> None:
    df = pd.read_parquet(TEST_PATH)
    X_test, y_test = df["text"], df["sentiment"]

    pipeline = joblib.load(MODEL_PATH)  # ja vem com o TF-IDF e o classificador juntos
    preds = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    # average=None devolve um valor por classe (em vez de uma media so) - importante
    # pra ver se o modelo erra mais pra um lado que pro outro
    precision = precision_score(y_test, preds, average=None, labels=LABELS)
    recall = recall_score(y_test, preds, average=None, labels=LABELS)
    f1 = f1_score(y_test, preds, average=None, labels=LABELS)
    f1_macro = f1_score(y_test, preds, average="macro")

    print("Avaliacao final (test.parquet - dado nunca visto pelo modelo ate agora):")
    print(f"  accuracy geral: {accuracy:.3f}")
    print(f"  f1_macro:       {f1_macro:.3f}\n")
    print(classification_report(y_test, preds, labels=LABELS))

    # Matriz de confusao: separa os dois tipos de erro possiveis (negativo
    # classificado como positivo, e vice-versa) - o numero mais importante pro
    # negocio, ja que os dois erros nao custam a mesma coisa (deixar passar uma
    # reclamacao como se fosse elogio esconde insatisfacao de cliente)
    cm = confusion_matrix(y_test, preds, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão — test.parquet")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Matriz de confusão salva em {CONFUSION_MATRIX_PATH}")

    # Salva tudo num JSON, do mesmo jeito que o train.py ja faz com
    # train_metrics.json - senao o numero so existe no terminal enquanto o
    # script roda
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": len(df),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist(),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em {METRICS_PATH}")


if __name__ == "__main__":
    main()
