
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

pacote = joblib.load("models/modelo_arroz_xgboost_2011_2020.pkl")
pipeline = pacote["pipeline"]

app = FastAPI(title="Previsão da Produção de Arroz")

class Entrada(BaseModel):

    Pais: str
    Year: int

    area_harvested: float
    yield_kg_ha: float

    temp_media_c: float
    precipitacao_mm: float

@app.post("/prever")
def prever(dados: Entrada):

    entrada = pd.DataFrame([{
        "Pais": dados.Pais,
        "Year": dados.Year,
        "Area harvested (ha)": dados.area_harvested,
        "Yield (kg/ha)": dados.yield_kg_ha,
        "Temp_Media_C": dados.temp_media_c,
        "Precipitacao_mm": dados.precipitacao_mm
    }])

    pred = pipeline.predict(entrada)[0]

    return {
        "pais": dados.Pais,
        "ano": dados.Year,
        "producao_prevista_t": round(float(pred), 2)
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
