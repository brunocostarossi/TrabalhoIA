# ATENÇÃO: este passo usa os rótulos ORIGINAIS do CSV, antes da descoberta do
# passo 7 (a 1ª foto do período de compra traz o vencedor do round anterior).
# As notas aqui ficam ~3 pontos acima do resultado final. Ver README.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("csgo_round_snapshots.csv")

# --- 1. Preparação: o modelo só entende números ---
# Alvo: CT = 1, T = 0
y = (df["round_winner"] == "CT").astype(int)
X = df.drop(columns=["round_winner"])

# bomb_planted: True/False -> 1/0
X["bomb_planted"] = X["bomb_planted"].astype(int)

# map: texto -> uma coluna 0/1 para cada mapa (one-hot encoding)
X = pd.get_dummies(X, columns=["map"], dtype=int)

print("Formato final de X:", X.shape)

# --- 2. Separação treino/teste ---
# shuffle=False: o CSV está em ordem cronológica e cada round aparece
# várias vezes. Embaralhar colocaria "fotos" do mesmo round no treino e
# no teste, inflando a acurácia (vazamento de dados).
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print("Treino:", X_treino.shape, "| Teste:", X_teste.shape)
print()

# --- 3. Baseline burro: sempre chuta a classe mais comum ---
burro = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
print("Acurácia do chute fixo:", round(accuracy_score(y_teste, burro.predict(X_teste)), 4))

# --- 4. Primeiro modelo de verdade: Regressão Logística ---
# StandardScaler coloca as colunas na mesma escala (dinheiro vai até 80000,
# jogadores vivos até 5), o que a regressão logística precisa.
logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logistica.fit(X_treino, y_treino)
previsoes = logistica.predict(X_teste)

print("Acurácia da Regressão Logística:", round(accuracy_score(y_teste, previsoes), 4))
print()
print(classification_report(y_teste, previsoes, target_names=["T", "CT"]))
