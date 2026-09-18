"""
Etapa 8 (Teste local): testes automatizados da API - garante que o /predict
responde corretamente pra casos claros de positivo/negativo, e que a API se
comporta bem em casos de borda (texto vazio, campo faltando).

Usa o TestClient do FastAPI: sobe a aplicacao direto em memoria, sem precisar
que a API esteja rodando numa porta antes (nao precisa de 'uvicorn' de pe pra
esses testes funcionarem).

Uso:
    pytest tests/test_api.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# Frases claramente positivas/negativas - nao sao os reviews reais do dataset,
# so exemplos escritos a mao pra checar que o modelo distingue os dois casos
# obvios. Nao substitui a avaliacao de verdade (evaluate.py, com o test.parquet) -
# isso aqui e um teste de "a API nao esta quebrada", nao uma medicao de qualidade.
POSITIVE_EXAMPLES = [
    "This product is amazing, I love it, best purchase ever",
    "Excellent quality, works perfectly, highly recommend to everyone",
    "Great value for the price, very satisfied with this purchase",
]

NEGATIVE_EXAMPLES = [
    "Terrible product, broke after one day, complete waste of money",
    "Very disappointed, poor quality, would not recommend to anyone",
    "Awful experience, arrived broken and customer service was rude",
]


@pytest.mark.parametrize("text", POSITIVE_EXAMPLES)
def test_predict_positive_examples(text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.parametrize("text", NEGATIVE_EXAMPLES)
def test_predict_negative_examples(text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_empty_text_is_rejected():
    # min_length=1 no PredictRequest deve barrar isso antes de chegar no modelo
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
