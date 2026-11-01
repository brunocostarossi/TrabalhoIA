import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from preparacao import carregar_dados

# Treina só no treino: a calibração tem que ser medida em dados nunca vistos
X_treino, X_teste, y_treino, y_teste = carregar_dados(novos_atributos=True, corrigir_rotulos=True)
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
prob_ct = modelo.predict_proba(X_teste)[:, 1]

# --- 1. Tabela de calibração ---
# Separa as previsões em faixas de 10% e compara: o que o modelo disse x o que aconteceu
faixas = pd.cut(prob_ct, bins=np.linspace(0, 1, 11), include_lowest=True)
tabela = pd.DataFrame({"previsto": prob_ct, "real": y_teste.values, "faixa": faixas})
tabela = tabela.groupby("faixa", observed=True).agg(
    previsto=("previsto", "mean"), real=("real", "mean"), fotos=("real", "size"))
tabela["erro"] = tabela["real"] - tabela["previsto"]
print("Faixa do modelo   disse   aconteceu   erro   fotos")
for faixa, l in tabela.iterrows():
    print(f"{str(faixa):16s} {l.previsto:6.1%}   {l.real:8.1%}  {l.erro:+6.1%}  {l.fotos:6.0f}")

# --- 2. Brier score: erro quadrático médio das probabilidades (0 = perfeito) ---
# Referência: chutar sempre a taxa de vitória do CT no treino
brier_modelo = brier_score_loss(y_teste, prob_ct)
brier_chute = brier_score_loss(y_teste, np.full(len(y_teste), y_treino.mean()))
print(f"\nBrier do modelo: {brier_modelo:.4f}  |  Brier do chute fixo: {brier_chute:.4f}")

# --- 3. Gráfico de confiabilidade ---
AZUL, TINTA, TINTA_2, GRADE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#fcfcfb")
for a in (ax, ax_hist):
    a.set_facecolor("#fcfcfb")
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(TINTA_2)
    a.tick_params(colors=TINTA_2)
    a.grid(color=GRADE, linewidth=0.8)
    a.set_axisbelow(True)

ax.plot([0, 1], [0, 1], linestyle="--", color=TINTA_2, linewidth=1.2)
ax.text(0.62, 0.55, "calibração perfeita", color=TINTA_2, rotation=40, fontsize=9)
ax.plot(tabela["previsto"], tabela["real"], color=AZUL, linewidth=2,
        marker="o", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2)
ax.set_ylabel("Fração em que o CT venceu", color=TINTA)
ax.set_title("Quando o modelo diz X%, o CT vence X%?", color=TINTA, loc="left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax_hist.hist(prob_ct, bins=np.linspace(0, 1, 21), color=AZUL, edgecolor="#fcfcfb", linewidth=2)
ax_hist.set_xlabel("Probabilidade de vitória do CT prevista pelo modelo", color=TINTA)
ax_hist.set_ylabel("Fotos", color=TINTA)

fig.tight_layout()
os.makedirs("graficos", exist_ok=True)
fig.savefig("graficos/calibracao.png", dpi=150)
print("Gráfico salvo em graficos/calibracao.png")
