# -*- coding: utf-8 -*-
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Agente Preditivo de Desempenho Acadêmico",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Agente Preditivo de Desempenho Acadêmico")
st.markdown("Preencha os dados do estudante abaixo para obter a predição do modelo e a análise do agente inteligente.")
st.divider()


st.subheader("📋 Dados do Estudante")

col1, col2 = st.columns(2)

with col1:
    gaming_hours    = st.number_input("Horas jogando por dia",    min_value=0.0, step=0.5)
    study_hours     = st.number_input("Horas estudando por dia",  min_value=0.0, step=0.5)
    sleep_hours     = st.number_input("Horas de sono por noite",  min_value=0.0, step=0.5)
    attendance      = st.number_input("Frequência nas aulas (%)", min_value=0.0, max_value=100.0, step=1.0)
    social_activity = st.number_input("Atividade social (0-10)",  min_value=0.0, max_value=10.0,  step=0.5)

with col2:
    device_usage     = st.number_input("Uso de dispositivos (h/dia)",  min_value=0.0, step=0.5)
    reaction_time_ms = st.number_input("Tempo de reação (ms)",         min_value=0.0, step=10.0)
    addiction_score  = st.number_input("Score de vício em jogos (0-10)", min_value=0.0, max_value=10.0, step=0.5)
    gender           = st.selectbox("Gênero", ["Male", "Female", "Other"])
    gaming_genre     = st.selectbox("Gênero de jogo preferido", ["Action", "RPG", "Sports", "Strategy", "Simulation"])

stress_level = st.select_slider("Nível de estresse", options=["Low", "Medium", "High"])

st.divider()


if st.button("🔍 Analisar", use_container_width=True, type="primary"):
    payload = {
        "gaming_hours":    gaming_hours,
        "study_hours":     study_hours,
        "sleep_hours":     sleep_hours,
        "attendance":      attendance,
        "social_activity": social_activity,
        "device_usage":    device_usage,
        "reaction_time_ms":reaction_time_ms,
        "addiction_score": addiction_score,
        "gender":          gender,
        "gaming_genre":    gaming_genre,
        "stress_level":    stress_level,
    }

    with st.spinner("Consultando modelo e agente inteligente..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar à API. Verifique se o servidor está rodando (`uvicorn api:app --reload`).")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("A requisição demorou demais. Tente novamente.")
            st.stop()
        except Exception as e:
            st.error(f"Erro inesperado: {e}")
            st.stop()

    # resultado bruto do modelo
    st.subheader("📊 Resultado do Modelo")
    aprovado = data["predicao"] == "Aprovado"

    col_res, col_prob = st.columns(2)
    with col_res:
        if aprovado:
            st.success(f"✅ {data['predicao']}")
        else:
            st.error(f"❌ {data['predicao']}")
    with col_prob:
        st.metric("Probabilidade de Aprovação", f"{data['prob_aprovado_pct']}%")

    st.progress(data["prob_aprovado_pct"] / 100)

    # explicação do agente
    st.divider()
    st.subheader("🤖 Análise do Agente Inteligente")
    st.markdown(data["explicacao"])
