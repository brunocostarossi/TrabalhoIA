import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from preparacao import carregar_dados

X_treino, X_teste, y_treino, y_teste = carregar_dados()
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
acc_base = accuracy_score(y_teste, modelo.predict(X_teste))
print(f"Acurácia sem embaralhar nada: {acc_base:.4f}\n")

# Agrupa as colunas por "conceito" (as 103 colunas viram 11 ideias)
colunas = X_teste.columns
grupos = {
    "vida": ["ct_health", "t_health"],
    "jogadores vivos": ["ct_players_alive", "t_players_alive"],
    "bomba plantada": ["bomb_planted"],
    "tempo restante": ["time_left"],
    "armadura + capacete": ["ct_armor", "t_armor", "ct_helmets", "t_helmets"],
    "armas": [c for c in colunas if "_weapon_" in c],
    "granadas": [c for c in colunas if "_grenade_" in c],
    "kits de defuse": ["ct_defuse_kits"],
    "dinheiro": ["ct_money", "t_money"],
    "placar": ["ct_score", "t_score"],
    "mapa": [c for c in colunas if c.startswith("map_")],
}

# Importância por permutação: embaralha o grupo inteiro e mede a queda
rng = np.random.default_rng(42)
resultados = {}
for nome, cols in grupos.items():
    quedas = []
    for _ in range(5):  # repete 5 vezes para reduzir o efeito do acaso
        X_embaralhado = X_teste.copy()
        ordem = rng.permutation(len(X_teste))
        X_embaralhado[cols] = X_teste[cols].values[ordem]
        quedas.append(acc_base - accuracy_score(y_teste, modelo.predict(X_embaralhado)))
    resultados[nome] = np.mean(quedas)

for nome, queda in sorted(resultados.items(), key=lambda kv: -kv[1]):
    print(f"{nome:22s} queda de {queda * 100:5.2f} pontos  {'#' * int(queda * 400)}")
