# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

df = pd.read_csv("Gaming_Academic_Performance.csv")

# Pré-processamento

df.drop(columns=["student_id"], inplace=True)

df["grades"] = df["grades"].clip(0, 100)
df["addiction_score"] = df["addiction_score"].clip(lower=0)

le = LabelEncoder()
for col in ["gender", "gaming_genre", "stress_level"]:
    df[col + "_enc"] = le.fit_transform(df[col])
 
df["aprovado"] = (df["grades"] >= 60).astype(int)

# Divisão dos dados Treino / Teste

features = [
    "gaming_hours", "study_hours", "sleep_hours", "attendance",
    "social_activity", "device_usage", "reaction_time_ms",
    "addiction_score", "gender_enc", "gaming_genre_enc", "stress_level_enc",
]
 
X = df[features]
y = df["aprovado"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
 
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)
 
# Modelo MLP

model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    max_iter=500,
    random_state=42,
)
model.fit(X_train_sc, y_train)
pred = model.predict(X_test_sc)
 
print("Relatório de classificação:")
print(classification_report(y_test, pred, target_names=["Reprovado", "Aprovado"]))
 
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
 
print("Métricas:")
print(f"  Acurácia:       {(tp + tn) / cm.sum():.4f}")
print(f"  Sensibilidade:  {tp / (tp + fn):.4f}")
print(f"  Especificidade: {tn / (tn + fp):.4f}")
print(f"  Precisão:       {tp / (tp + fp):.4f}")

joblib.dump(model,  "melhor_modelo_mlp.pkl")
joblib.dump(scaler,        "scaler.pkl")
print("\nModelo exportado: melhor_modelo_mlp.pkl | scaler.pkl")