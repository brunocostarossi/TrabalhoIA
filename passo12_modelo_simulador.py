"""Exporta o modelo do simulador (interface/simulador.html) para JavaScript.

O simulador pede ao usuário só o que aparece na tabela, sem dinheiro. Por isso
este modelo é treinado SEM as colunas de dinheiro (ct_money, t_money,
dif_dinheiro) e sem os atributos de "momento" (que dependem da foto anterior,
que não existe num cenário montado à mão).

As árvores do HistGradientBoostingClassifier são salvas em um .js e avaliadas
no navegador: probabilidade = sigmoide(base + soma dos valores das folhas).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from preparacao import (PRECO_CAPACETE, PRECO_COLETE, PRECO_KIT, PRECOS,
                        carregar_dados)

SEM_DINHEIRO = ["ct_money", "t_money", "dif_dinheiro"]
SAIDA = os.path.join("interface", "modelo_simulador.js")
AMOSTRAS = os.environ.get("AMOSTRAS_VERIFICACAO")  # opcional: casos para conferir o JS

X_treino, X_teste, y_treino, y_teste = carregar_dados(novos_atributos=True, corrigir_rotulos=True)
X_treino, X_teste = X_treino.drop(columns=SEM_DINHEIRO), X_teste.drop(columns=SEM_DINHEIRO)

# --- 1. Quanto se perde sem dinheiro e sem momento? (teste, uma vez) ---
avaliacao = HistGradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
acertos = avaliacao.predict(X_teste) == y_teste
print(f"Teste sem dinheiro: {acertos.mean():.4f}")
print(f"  período de compra: {acertos[X_teste['time_left'] > 170].mean():.4f}")
print(f"  round em andamento: {acertos[X_teste['time_left'] <= 170].mean():.4f}")

# --- 2. Modelo final do simulador: treina com tudo, como no passo 6 ---
X_tudo, y_tudo = pd.concat([X_treino, X_teste]), pd.concat([y_treino, y_teste])
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_tudo, y_tudo)


def arvore(preditor):
    nos = preditor.nodes
    assert not nos["is_categorical"].any()
    return {
        "f": nos["feature_idx"].tolist(),
        "t": [float(x) for x in nos["num_threshold"]],
        "e": nos["left"].tolist(),            # filho da esquerda (valor <= limiar)
        "d": nos["right"].tolist(),           # filho da direita
        "m": nos["missing_go_to_left"].astype(int).tolist(),
        "v": [round(float(x), 7) for x in nos["value"]],
        "folha": nos["is_leaf"].astype(int).tolist(),
    }


exportado = {
    "colunas": list(X_tudo.columns),
    "base": float(np.ravel(modelo._baseline_prediction)[0]),
    "arvores": [arvore(iteracao[0]) for iteracao in modelo._predictors],
    "precos": PRECOS,
    "preco_colete": PRECO_COLETE,
    "preco_capacete": PRECO_CAPACETE,
    "preco_kit": PRECO_KIT,
}
os.makedirs("interface", exist_ok=True)
with open(SAIDA, "w", encoding="utf-8") as arquivo:
    arquivo.write("// Gerado por passo12_modelo_simulador.py. Não editar à mão.\n")
    arquivo.write("window.MODELO_SIMULADOR = ")
    json.dump(exportado, arquivo, separators=(",", ":"))
    arquivo.write(";\n")
print(f"{len(exportado['arvores'])} árvores salvas em {SAIDA}")

# --- 3. Casos para conferir que o JavaScript dá o mesmo resultado ---
if AMOSTRAS:
    linhas = X_teste.sample(300, random_state=1)
    casos = []
    for _, x in linhas.iterrows():
        def time(lado):
            itens = {}
            for coluna, valor in x.items():
                for tipo in ("weapon", "grenade"):
                    prefixo = f"{lado}_{tipo}_"
                    if coluna.startswith(prefixo) and valor > 0:
                        itens[coluna[len(prefixo):]] = int(valor)
            dados = {"vivos": int(x[f"{lado}_players_alive"]), "vida": int(x[f"{lado}_health"]),
                     "colete": int(x[f"{lado}_armor"]), "capacetes": int(x[f"{lado}_helmets"]),
                     "itens": itens}
            if lado == "ct":
                dados["kits"] = int(x["ct_defuse_kits"])
            return dados
        mapa = next(c[len("map_"):] for c in X_teste.columns if c.startswith("map_") and x[c] == 1)
        casos.append({
            "estado": {"mapa": mapa, "placar_ct": int(x["ct_score"]), "placar_t": int(x["t_score"]),
                       "tempo": float(x["time_left"]), "bomba": bool(x["bomb_planted"]),
                       "ct": time("ct"), "t": time("t")},
            "esperado": float(modelo.predict_proba(x.to_frame().T.astype(float))[0, 1]),
        })
    with open(AMOSTRAS, "w", encoding="utf-8") as arquivo:
        arquivo.write("window.CASOS = ")
        json.dump(casos, arquivo)
        arquivo.write(";\n")
    print(f"{len(casos)} casos de verificação salvos em {AMOSTRAS}")
