# -*- coding: utf-8 -*-
import os
import numpy as np
import joblib
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# configs

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Chave da API não encontrada")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

model = joblib.load("melhor_modelo_mlp.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Agente Preditivo de Desempenho Acadêmico",
    description="Prediz aprovação/reprovação de estudantes e explica o resultado via Gemini",
    version="1.0.0",
)

# schemas de entrada

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
GENRE_MAP = {"Action": 0, "RPG": 1, "Sports": 2, "Strategy": 3, "Simulation": 4}
STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}

class StudentData(BaseModel):
    gaming_hours: float = Field(..., ge=0, description="Horas diárias jogando")
    study_hours: float = Field(..., ge=0, description="Horas diárias estudando")
    sleep_hours: float = Field(..., ge=0, description="Horas de sono por noite")
    attendance: float = Field(..., ge=0, description="Frequência nas aulas (%)")
    social_activity: float = Field(..., ge=0, description="Nível de atividade social (0-10)")
    device_usage: float = Field(..., ge=0, description="Horas díarias de uso de dispositivos")
    reaction_time_ms: float = Field(..., ge=0, description="Tempo de reação em ms")
    addiction_score: float = Field(..., ge=0, description="Score de vício em jogos (0-10)")
    gender: str = Field(..., description="Male | Female | Other")
    gaming_genre: str = Field(..., description="Action | RPG | Sports | Strategy | Simulation")
    stress_level: str = Field(..., description="Low | Medium | High")

# prompt
SYSTEM_PROMPT = """
Você é um assistente especializado em análise de desempenho acadêmico de estudantes.
Você receberá os dados de um estudante e o resultado de um modelo preditivo de Machine Learning (MLP).

Suas instruções:
- Explique de forma clara e empática se o estudante tente a ser APROVADO ou REPROVADO.
- Justifique a predição com base nos dados fornecidos, destacando os fatores mais relevantes.
- Aponte os pontos de atenção e sugira melhorias concretas quando o aluno estiver em risco.
- Use linguagem acessível, sem jargões técnicos de ML.
- Seja direto e objetivo. Não invente informações além dos dados fornecidos.
- Nunca afirme certeza absoluta; trate a predição como uma estimativa probabilística.
""".strip()

# endpoint de predição
@app.post("/predict")
def predict(data: StudentData):
    if data.gender not in GENDER_MAP:
        raise HTTPException(400, f"gender inválido. Use: {list(GENDER_MAP.keys())}")
    if data.gaming_genre not in GENRE_MAP:
        raise HTTPException(400, f"gaming_genre inválido. Use: {list(GENRE_MAP.keys())}")
    if data.stress_level not in STRESS_MAP:
        raise HTTPException(400, f"stress_level inválido. Use: {list(STRESS_MAP.keys())}")
    
    features = np.array([[
        data.gaming_hours,
        data.study_hours,
        data.sleep_hours,
        data.attendance,
        data.social_activity,
        data.device_usage,
        data.reaction_time_ms,
        data.addiction_score,
        GENDER_MAP[data.gender],
        GENRE_MAP[data.gaming_genre],
        STRESS_MAP[data.stress_level],
    ]])

    features_sc = scaler.transform(features)
    prediction = int(model.predict(features_sc)[0])
    probabilities = model.predict_proba(features_sc)[0]
    prob_aprovado = round(float(probabilities[1]) * 100, 1)
    resultado = "Aprovado" if prediction == 1 else "Reprovado"

    prompt = f"""
{SYSTEM_PROMPT}

Dados do estudante:
- Horas jogando por dia: {data.gaming_hours}h
- Horas estudando por dia: {data.study_hours}h
- Horas de sono por noite: {data.sleep_hours}h
- Frequência nas aulas: {data.attendance}%
- Atividade social: {data.social_activity}/10
- Uso de dispositivos: {data.device_usage}h/dia
- Tempo de reação: {data.reaction_time_ms}ms
- Score de vício em jogos: {data.addiction_score}/10
- Gênero: {data.gender}
- Gênero de jogo preferido: {data.gaming_genre}
- Nível de estresse: {data.stress_level}
 
Resultado do modelo preditivo: {resultado} (probabilidade de aprovação: {prob_aprovado}%)
 
Com base nesses dados, forneça sua análise.
""".strip()
    
    try:
        response = client.models.generate_content(
            model = GEMINI_MODEL,
            contents= prompt,
        )
        explicacao = response.text
    except Exception as e:
        raise HTTPException(500, f"Erro ao consultar o Gemini: {str(e)}")
    
    return {
        "predicao": resultado,
        "prob_aprovado_pct": prob_aprovado,
        "explicacao": explicacao 
    }

@app.get("/")
def health():
    return {
        "status": "ok",
        "modelo": "MLP (64,) relu",
        "llm": "gemini-2.5-flash"
    }