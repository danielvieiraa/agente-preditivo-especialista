# Agente Preditivo de Desempenho Acadêmico

Sistema de Machine Learning integrado com IA generativa para prever o desempenho acadêmico de estudantes com base em seus hábitos de uso de jogos eletrônicos.

---

## Visão Geral

O projeto utiliza um modelo MLP (Multi-Layer Perceptron) treinado sobre o dataset *Gaming Academic Performance* para classificar estudantes como **Aprovados** ou **Reprovados**. A predição é então interpretada pelo modelo **Gemini 2.5 Flash** (Google AI), que gera uma explicação clara e fundamentada ao usuário.

---

## Estrutura do Projeto

```
agente-preditivo-especialista/
├── Gaming_Academic_Performance.csv  # dataset
├── mlp.py                           # treinamento e exportação do modelo
├── api.py                           # backend FastAPI + integração Gemini
├── melhor_modelo_mlp.pkl            # modelo treinado (gerado pelo mlp.py)
├── scaler.pkl                       # normalizador (gerado pelo mlp.py)
├── .env                             # chave da API (não versionado)
├── .gitignore
└── LEIAME.md
```

---

## Pré-requisitos

- Python 3.10+
- Conta no [Google AI Studio](https://aistudio.google.com/) para obter a chave da API Gemini (plano gratuito disponível)

---

## Instalação

**1. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/agente-preditivo-especialista.git
cd agente-preditivo-especialista
```

**2. Instale as dependências:**
```bash
pip install pandas scikit-learn matplotlib joblib fastapi uvicorn google-genai python-dotenv
```

**3. Configure a chave da API:**

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
```
GEMINI_API_KEY=sua_chave_aqui
```

---

## Como Rodar

**Passo 1 — Treinar o modelo e gerar os arquivos `.pkl`:**
```bash
python mlp.py
```
Isso irá treinar o MLP, exibir as métricas de avaliação e salvar `melhor_modelo_mlp.pkl` e `scaler.pkl` na pasta do projeto.

**Passo 2 — Subir o servidor backend:**
```bash
uvicorn api:app --reload
```
O servidor estará disponível em `http://127.0.0.1:8000`.

**Passo 3 — Testar a API:**

Acesse `http://127.0.0.1:8000/docs` no navegador para abrir a interface interativa (Swagger UI).

Exemplo de requisição para o endpoint `POST /predict`:
```json
{
  "gaming_hours": 1,
  "study_hours": 5,
  "sleep_hours": 8,
  "attendance": 95,
  "social_activity": 7,
  "device_usage": 2,
  "reaction_time_ms": 250,
  "addiction_score": 1,
  "gender": "Female",
  "gaming_genre": "Strategy",
  "stress_level": "Low"
}
```

Campos aceitos:

| Campo | Tipo | Valores aceitos |
|---|---|---|
| `gaming_hours` | float | >= 0 |
| `study_hours` | float | >= 0 |
| `sleep_hours` | float | >= 0 |
| `attendance` | float | 0 a 100 |
| `social_activity` | float | >= 0 |
| `device_usage` | float | >= 0 |
| `reaction_time_ms` | float | >= 0 |
| `addiction_score` | float | >= 0 |
| `gender` | string | `Male`, `Female`, `Other` |
| `gaming_genre` | string | `Action`, `RPG`, `Sports`, `Strategy`, `Simulation` |
| `stress_level` | string | `Low`, `Medium`, `High` |

---

## Tecnologias Utilizadas

- **Machine Learning:** Python, Scikit-learn (MLPClassifier), Pandas, Joblib
- **Backend:** FastAPI, Uvicorn, Pydantic
- **IA Generativa:** Google Gemini 2.5 Flash (`google-genai`)
- **Utilitários:** python-dotenv, Matplotlib

---

## Diário de Bordo de Contribuições

### Daniel Vieira

**Etapa A — Preparação e Modelagem de Machine Learning**

- Realizei o carregamento do dataset *Gaming Academic Performance* e o pré-processamento dos dados, incluindo remoção da coluna `student_id`, correção de valores fora do intervalo válido (`grades` e `addiction_score`) e codificação de variáveis categóricas (`gender`, `gaming_genre`, `stress_level`) com `LabelEncoder`.
- Defini a variável alvo binária `aprovado` (notas >= 60) e realizou a separação dos dados em treino (80%) e teste (20%) com estratificação e `StandardScaler` para normalização.
- Gerei os gráficos exploratórios exigidos: mapa de correlação entre variáveis numéricas, box plots e gráficos de distribuição de frequência com a biblioteca Seaborn.
- Avaliei e comparei quatro configurações de MLP (`hidden_layer_sizes` variando entre camadas simples e múltiplas, com ativações `relu` e `tanh`), selecionando a configuração `(64,) relu` como melhor modelo com acurácia de 93,31%.
- Calculei as métricas de avaliação manualmente a partir da matriz de confusão: acurácia, sensibilidade, especificidade e precisão.
- Exportei o modelo final e o normalizador via `joblib` (`melhor_modelo_mlp.pkl` e `scaler.pkl`).

**Etapa B — Backend e Agente Inteligente**

- Desenvolvi o servidor backend com **FastAPI**, expondo o endpoint `POST /predict` para receber dados de novos estudantes, aplicar o modelo treinado e retornar a predição com probabilidade de aprovação.
- Integrei a API do **Google Gemini 2.5 Flash** (`google-genai`) ao backend, implementando um System Prompt especializado que instrui o modelo a interpretar a predição de forma clara, empática e sem alucinações.
- Configurei o carregamento seguro da chave da API via arquivo `.env` com `python-dotenv`.
- Validei o funcionamento completo da API com casos de teste reais: perfil de risco (1,5% de probabilidade de aprovação) e perfil positivo (99,7% de probabilidade de aprovação), com explicações coerentes geradas pelo Gemini em ambos os casos.
- Migrei a biblioteca de integração com o Gemini de `google-generativeai` (descontinuada) para `google-genai`.
- Configurei o `.gitignore` para excluir `__pycache__/`, arquivos `.pyc` e o `.env` do versionamento.

---

### Guilherme Bottcher

*(a preencher)*

---