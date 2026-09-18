"""
Etapa 4 (Treino): le data/gold/train.parquet, treina o classificador de
sentimento e salva o modelo (ja com o vetorizador embutido) em models/.

Modelo escolhido: TF-IDF + Regressao Logistica.
Por que esse par, e nao algo mais complexo (ex: rede neural/transformer):
  - A EDA (notebooks/eda.ipynb) ja mostrou que existe sinal de vocabulario
    separando as classes (palavras e principalmente BIGRAMAS como "waste money",
    "highly recommend"), entao um modelo de bag-of-words ja capta boa parte
    do problema, sem precisar de algo caro computacionalmente.
  - TF-IDF pondera pela relevancia da palavra/bigrama (nao so pela frequencia
    bruta), o que a EDA mostrou ser necessario (palavras soltas mais frequentes
    se repetiam nas duas classes).
  - Regressao Logistica e rapida de treinar (mesmo com dataset grande), nao
    precisa de GPU, e da pra inspecionar os pesos aprendidos (quais palavras
    mais empurram pra positivo/negativo) - importante pra explicar o modelo
    pro time/apresentacao, nao so usar como caixa-preta.
  - E o par padrao pra classificacao de texto com poucos dados/tempo, e serve
    de baseline solido antes de considerar algo mais complexo (nao faz sentido
    comecar complexo sem antes saber se o simples ja resolve o problema).

Validacao cruzada (5 folds, estratificada) roda SOMENTE sobre o train.parquet,
nunca toca no test.parquet - serve pra estimar a performance de forma mais
robusta do que um unico treino, e pra checar se o modelo e estavel entre
fatias diferentes do treino. O TF-IDF fica DENTRO do Pipeline validado, pra
cada fold re-ajustar o vetorizador so com a fatia de treino daquele fold -
sem isso, haveria vazamento de dado entre os folds (o vocabulario "veria"
dado de validacao antes da hora).

Uso:
    python src/train.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "gold" / "train.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "train_metrics.json"

RANDOM_STATE = 42
N_FOLDS = 5


# max_features: limita o tamanho do vocabulario (evita explosao de memoria/overfitting
#   em termos raros quando o dataset ficar grande)
# ngram_range=(1, 2): usa palavra isolada E par de palavras (bigrama) - a EDA mostrou
#   que bigramas ("waste money", "highly recommend") diferenciam sentimento melhor
#   do que palavra sozinha
# min_df=2: ignora termos que aparecem em menos de 2 reviews (ruido raro)
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ])


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    X, y = df["text"], df["sentiment"]

    pipeline = build_pipeline()

    # StratifiedKFold mantem a proporcao de classes em cada fold (parecido com o
    # que ja conferimos manualmente no build_features.py, so que aplicado tambem
    # dentro da validacao)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro"])

    print(f"Validacao cruzada ({N_FOLDS} folds, so sobre o treino):")
    print(f"  accuracy: {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std():.3f})")
    print(f"  f1_macro: {scores['test_f1_macro'].mean():.3f} (+/- {scores['test_f1_macro'].std():.3f})")

    # Treino final: agora sim ajusta o pipeline (TF-IDF + modelo) com o treino INTEIRO -
    # a validacao cruzada acima foi so pra medir estabilidade, nao gera o modelo final
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Salva as metricas da validacao cruzada num arquivo, senao elas so existem no
    # terminal enquanto o script roda e se perdem depois - isso serve pra consultar
    # depois (ex: na apresentacao da Etapa 10) sem precisar rodar o treino de novo
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(df),
        "n_folds": N_FOLDS,
        "cv_accuracy_mean": round(scores["test_accuracy"].mean(), 4),
        "cv_accuracy_std": round(scores["test_accuracy"].std(), 4),
        "cv_f1_macro_mean": round(scores["test_f1_macro"].mean(), 4),
        "cv_f1_macro_std": round(scores["test_f1_macro"].std(), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nModelo salvo em {MODEL_PATH}")
    print(f"Metricas salvas em {METRICS_PATH}")


if __name__ == "__main__":
    main()