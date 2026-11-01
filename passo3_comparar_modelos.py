import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from preparacao import carregar_dados

X_treino, X_teste, y_treino, y_teste = carregar_dados()

modelos = {
    # Fronteira de decisão "reta": soma ponderada das variáveis
    "Regressão Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    # Muitas árvores independentes, cada uma vê uma amostra diferente; voto da maioria
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    # Árvores em sequência, cada uma corrige os erros da anterior
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}

for nome, modelo in modelos.items():
    inicio = time.time()
    modelo.fit(X_treino, y_treino)
    acc_treino = accuracy_score(y_treino, modelo.predict(X_treino))
    acc_teste = accuracy_score(y_teste, modelo.predict(X_teste))
    print(f"{nome:22s} treino={acc_treino:.4f}  teste={acc_teste:.4f}  ({time.time() - inicio:.1f}s)")
