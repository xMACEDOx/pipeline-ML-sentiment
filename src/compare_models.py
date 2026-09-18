"""
Etapa 4 (comparacao de modelos): treina e compara tres candidatos - Regressao
Logistica, XGBoost e uma rede neural simples (Keras/TensorFlow) - todos usando
a MESMA representacao de texto (TF-IDF), pra isolar exatamente o que muda
entre eles: o algoritmo, nao a forma de representar o texto. Se a gente
trocasse tambem a representacao (ex: usar embeddings so na rede neural), nao
daria pra saber se uma diferenca de resultado vem do algoritmo ou da
representacao - por isso os tres recebem o mesmo vetor de entrada.

Avaliacao: um UNICO split treino/validacao (estratificado, 80/20) tirado do
proprio train.parquet - nao e a validacao cruzada de 5 folds do train.py.
Motivo: treinar a rede neural 5 vezes (uma por fold, como seria numa CV
completa) e caro e lento, principalmente sem GPU - um unico split e o
suficiente pra comparar os tres modelos de forma justa (todos veem
exatamente o mesmo treino e a mesma validacao) sem gastar tempo demais.
O test.parquet continua completamente intocado, reservado pra avaliacao
final (Etapa 5), independente de qual modelo vencer aqui.

Resultado: imprime e salva em models/model_comparison.json as metricas dos
tres candidatos, pra decidir qual vira o modelo "oficial" do projeto
(o que hoje esta salvo/treinado pelo train.py).

Uso:
    python src/compare_models.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# silencia os logs bem verbosos do TensorFlow (infos de CPU/CUDA que nao sao erro
# de verdade - so ruido). Precisa ser setado ANTES de importar tensorflow/keras.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "gold" / "train.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

RANDOM_STATE = 42
# menor que os 20_000 do train.py: a rede neural precisa de matriz DENSA (nao
# esparsa como o sklearn usa por baixo dos panos) - com vocabulario grande isso
# estouraria memoria facil, entao aqui o limite e mais conservador
MAX_FEATURES = 5_000


def build_nn(input_dim: int):
    """Rede neural simples (MLP - perceptron multicamadas): uma camada escondida
    de 64 neuronios. E o suficiente pra ver se aprender combinacoes nao-lineares
    das features de TF-IDF ajuda, sem virar um projeto de deep learning por si
    so (nada de embeddings, camadas convolucionais/recorrentes, etc. - isso
    seria over-engineering pro escopo desse case)."""
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),  # zera 30% dos neuronios a cada passo de treino,
                                     # so durante o treino - reduz overfitting num
                                     # dataset pequeno
        keras.layers.Dense(1, activation="sigmoid"),  # sigmoid = probabilidade entre 0 e 1
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    y = df["sentiment"].map({"negative": 0, "positive": 1}).to_numpy()

    X_train_text, X_valid_text, y_train, y_valid = train_test_split(
        df["text"], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # TF-IDF ajustado SO com a fatia de treino desse split - o vetorizador nunca
    # ve o texto da validacao antes da hora, mesma disciplina do train.py/process.py
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_valid_vec = vectorizer.transform(X_valid_text)

    results = {}

    # --- Regressao Logistica (o baseline que ja tinhamos) ---
    logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    logreg.fit(X_train_vec, y_train)
    preds = logreg.predict(X_valid_vec)
    results["logistic_regression"] = {
        "accuracy": round(accuracy_score(y_valid, preds), 4),
        "f1_macro": round(f1_score(y_valid, preds, average="macro"), 4),
    }

    # --- XGBoost (arvores com boosting - captura interacoes entre termos que
    # um modelo linear como a Regressao Logistica nao capta) ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train_vec, y_train)
    preds = xgb.predict(X_valid_vec)
    results["xgboost"] = {
        "accuracy": round(accuracy_score(y_valid, preds), 4),
        "f1_macro": round(f1_score(y_valid, preds, average="macro"), 4),
    }

    # --- Rede neural simples (Keras/TensorFlow) ---
    nn = build_nn(input_dim=X_train_vec.shape[1])
    nn.fit(
        X_train_vec.toarray(), y_train,
        validation_data=(X_valid_vec.toarray(), y_valid),
        epochs=10,
        batch_size=32,
        verbose=0,
    )
    nn_probs = nn.predict(X_valid_vec.toarray(), verbose=0).ravel()
    nn_preds = (nn_probs > 0.5).astype(int)
    results["neural_network"] = {
        "accuracy": round(accuracy_score(y_valid, nn_preds), 4),
        "f1_macro": round(f1_score(y_valid, nn_preds, average="macro"), 4),
    }

    print("Comparacao de modelos (holdout 80/20 tirado do train.parquet, mesma vetorizacao TF-IDF pros tres):")
    for name, metrics in results.items():
        print(f"  {name:20s} accuracy={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(X_train_text),
        "n_valid_rows": len(X_valid_text),
        "max_features": MAX_FEATURES,
        "results": results,
    }
    COMPARISON_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nComparacao salva em {COMPARISON_PATH}")


if __name__ == "__main__":
    main()