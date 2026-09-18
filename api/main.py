"""
Etapa 7 (API): expoe o modelo treinado (models/sentiment_pipeline.joblib) como
uma API interna. Recebe o texto de um review, roda a predicao (o mesmo
Pipeline treinado no train.py - TF-IDF + Regressao Logistica) e devolve o
sentimento previsto e a confianca do modelo naquela previsao.

Uso:
    uvicorn api.main:app --reload

Depois, POST em http://127.0.0.1:8000/predict com body {"text": "..."}.
Docs interativas (Swagger) em http://127.0.0.1:8000/docs.
"""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_pipeline.joblib"

# Carrega o modelo UMA vez, quando a API sobe - nao a cada request (seria lento
# e desperdicaria recurso atoa). O carregamento e rapido e sincrono (e so um
# arquivo pequeno com o Pipeline ja treinado), entao nao precisa da
# complexidade de um lifespan assincrono aqui - carregar direto no import do
# modulo ja e suficiente pro tamanho desse projeto.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None  # a API ainda sobe, mas o /predict avisa que falta treinar o modelo


app = FastAPI(
    title="Review Sentiment API",
    description="Classifica o sentimento (positivo/negativo) de reviews de produtos.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto do review a ser classificado")


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float


@app.get("/health")
def health() -> dict:
    """Checagem simples de saude da API - o tipo de endpoint que um load
    balancer/orquestrador usa pra saber se o
    servico esta de pe e com o modelo carregado, antes de mandar trafego pra ele."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao encontrado em models/sentiment_pipeline.joblib - rode 'python src/train.py' antes de subir a API.",
        )

    # predict_proba devolve a probabilidade de cada classe (ex: [0.12, 0.88]);
    # pegamos a probabilidade da classe que foi de fato prevista, como "confianca"
    sentiment = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    class_index = list(model.classes_).index(sentiment)
    confidence = float(probabilities[class_index])

    return PredictResponse(sentiment=sentiment, confidence=confidence)
