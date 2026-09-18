"""
Etapa 3 (EDA rapida): olha o Bronze antes de limpar, so pra embasar as decisoes
de pre-processamento e confirmar que o dataset faz sentido. Nao gera arquivo novo.

Uso:
    python src/eda.py
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BRONZE_PATH = Path(__file__).resolve().parent.parent / "data" / "bronze" / "reviews_raw.csv"

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "is", "it", "this", "in", "for",
    "i", "was", "on", "with", "my", "that", "are", "have", "but", "so", "not",
    "you", "me", "as", "be", "at", "very",
}


def top_words(texts: pd.Series, n: int = 15) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for t in texts:
        words = re.findall(r"[a-z']+", str(t).lower())
        counter.update(w for w in words if w not in STOPWORDS and len(w) > 2)
    return counter.most_common(n)


def main() -> None:
    df = pd.read_csv(BRONZE_PATH)

    print("Linhas:", len(df))
    print("\nBalanceamento de classes (0=negativo, 1=positivo):\n", df["label"].value_counts())
    print("\nNulos por coluna:\n", df.isna().sum())
    print("Duplicatas (content):", df["content"].astype(str).duplicated().sum())

    df["len_chars"] = df["content"].astype(str).str.len()
    print("\nTamanho do review (caracteres):\n", df["len_chars"].describe())

    print("\nTop palavras - negativos:", top_words(df.loc[df["label"] == 0, "content"]))
    print("\nTop palavras - positivos:", top_words(df.loc[df["label"] == 1, "content"]))


if __name__ == "__main__":
    main()
