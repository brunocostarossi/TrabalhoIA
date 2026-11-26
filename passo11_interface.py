"""Gera os dados da interface de placar (interface/placar.html).

Treina o modelo final (Gradient Boosting + novos atributos + momento, rótulos
corrigidos) só com o TREINO e calcula a probabilidade de vitória em rodadas do
TESTE, que o modelo nunca viu. Exporta uma amostra dessas rodadas em um arquivo
.js que a página HTML lê direto do disco (sem precisar de servidor).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from preparacao import carregar_dados, limpar_rotulos, marcar_rounds

QTD_RODADAS = 150  # quantas rodadas do teste vão para a interface
SAIDA = os.path.join("interface", "placar_dados.js")

# --- 1. Modelo final treinado só no treino; probabilidades no teste ---
X_treino, X_teste, y_treino, y_teste = carregar_dados(
    novos_atributos=True, corrigir_rotulos=True, atributos_momento=True)
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
prob_ct = modelo.predict_proba(X_teste)[:, 1]
print(f"Acurácia no teste: {(modelo.predict(X_teste) == y_teste).mean():.4f}")

# --- 2. As mesmas linhas do teste, mas com os valores originais do CSV ---
# carregar_dados faz marcar_rounds -> limpar_rotulos e mantém o índice,
# então o índice de X_teste aponta para as linhas certas de df.
df = limpar_rotulos(marcar_rounds(pd.read_csv("csgo_round_snapshots.csv")))
teste = df.loc[X_teste.index].copy()
teste["prob_ct"] = prob_ct


def equipamento(linha, lado):
    """Armas e granadas do time, só as que existem (quantidade > 0)."""
    itens = {}
    for tipo in ("weapon", "grenade"):
        prefixo = f"{lado}_{tipo}_"
        for coluna in linha.index:
            if coluna.startswith(prefixo) and linha[coluna] > 0:
                itens[coluna[len(prefixo):]] = int(linha[coluna])
    return itens


def time(linha, lado):
    dados = {
        "vivos": int(linha[f"{lado}_players_alive"]),
        "vida": int(linha[f"{lado}_health"]),
        "colete": int(linha[f"{lado}_armor"]),
        "capacetes": int(linha[f"{lado}_helmets"]),
        "itens": equipamento(linha, lado),
    }
    if lado == "ct":
        dados["kits"] = int(linha["ct_defuse_kits"])
    return dados


# --- 3. Amostra de rodadas espalhadas pelo teste ---
# A 1ª rodada do teste pode estar cortada (começou no treino), então fica de fora.
ids = teste["round_id"].unique()[1:]
fotos_por_rodada = teste.groupby("round_id").size()
ids = [i for i in ids if fotos_por_rodada[i] >= 3]  # rodadas com alguma história
escolhidas = [ids[i] for i in np.linspace(0, len(ids) - 1, QTD_RODADAS).astype(int)]

rodadas = []
for rid in escolhidas:
    fotos = teste[teste["round_id"] == rid]
    primeira = fotos.iloc[0]
    rodadas.append({
        "mapa": primeira["map"],
        "placar_ct": int(primeira["ct_score"]),
        "placar_t": int(primeira["t_score"]),
        "fotos": [{
            "tempo": round(float(f["time_left"]), 2),
            "bomba": bool(f["bomb_planted"]),
            "prob_ct": round(float(f["prob_ct"]), 4),
            "ct": time(f, "ct"),
            "t": time(f, "t"),
        } for _, f in fotos.iterrows()],
    })

os.makedirs("interface", exist_ok=True)
with open(SAIDA, "w", encoding="utf-8") as arquivo:
    arquivo.write("// Gerado por passo11_interface.py. Não editar à mão.\n")
    arquivo.write("window.DADOS_PLACAR = ")
    json.dump(rodadas, arquivo, ensure_ascii=False, separators=(",", ":"))
    arquivo.write(";\n")
print(f"{len(rodadas)} rodadas salvas em {SAIDA}")
