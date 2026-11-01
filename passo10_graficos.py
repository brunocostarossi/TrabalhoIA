import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from preparacao import carregar_dados
from passo6_demonstracao import probabilidade_ct

os.makedirs("graficos", exist_ok=True)
FUNDO, TINTA, TINTA_2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AZUL, CINZA = "#2a78d6", "#b5b4ae"


def novo_grafico(titulo, tamanho=(7, 4.2)):
    fig, ax = plt.subplots(figsize=tamanho)
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(titulo, color=TINTA, loc="left")
    return fig, ax


def rotular_barras(ax, barras, formato="{:.1f}%"):
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1.2,
                formato.format(barra.get_height()), ha="center", color=TINTA, fontsize=10)


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(f"graficos/{nome}", dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em graficos/{nome}")


def gradient_boosting():
    return HistGradientBoostingClassifier(random_state=42)


# --- 1. Linha do tempo da acurácia: cada queda é um problema corrigido ---
X_tr, X_te, y_tr, y_te = carregar_dados(novos_atributos=True)
X_tudo, y_tudo = pd.concat([X_tr, X_te]), pd.concat([y_tr, y_te])
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_tudo, y_tudo, test_size=0.2, shuffle=True, random_state=42)
acc_aleatoria = accuracy_score(ya_te, gradient_boosting().fit(Xa_tr, ya_tr).predict(Xa_te))
acc_ordem = accuracy_score(y_te, gradient_boosting().fit(X_tr, y_tr).predict(X_te))

X_tr, X_te, y_tr, y_te = carregar_dados(novos_atributos=True, corrigir_rotulos=True, atributos_momento=True)
modelo_final = gradient_boosting().fit(X_tr, y_tr)
acertos = modelo_final.predict(X_te) == y_te
acc_final = acertos.mean()

fig, ax = novo_grafico("Cada queda na acurácia é um problema corrigido")
etapas = ["Divisão aleatória\n(vazamento)", "Divisão em ordem\n(rótulos com erro)", "Rótulos corrigidos\n(resultado real)"]
barras = ax.bar(etapas, [acc_aleatoria * 100, acc_ordem * 100, acc_final * 100],
                color=[CINZA, CINZA, AZUL], width=0.55)
rotular_barras(ax, barras)
ax.axhline(50, color=TINTA_2, linestyle="--", linewidth=1.2)
ax.text(-0.45, 51.5, "chute: 50%", color=TINTA_2, fontsize=9, ha="left")
ax.set_ylim(0, 100)
ax.set_ylabel("Acurácia no teste (%)", color=TINTA)
salvar(fig, "linha_do_tempo_acuracia.png")

# --- 2. Acurácia por vantagem de jogadores vivos ---
vantagem = (X_te["ct_players_alive"] - X_te["t_players_alive"]).clip(-3, 3)
por_vantagem = acertos.groupby(vantagem).mean() * 100
fig, ax = novo_grafico("Com vantagem numérica, o modelo quase não erra")
rotulos = ["T +3\nou mais", "T +2", "T +1", "Igual", "CT +1", "CT +2", "CT +3\nou mais"]
barras = ax.bar(rotulos, por_vantagem.values, color=AZUL, width=0.6)
rotular_barras(ax, barras, "{:.0f}%")
ax.set_ylim(0, 110)
ax.set_ylabel("Acurácia no teste (%)", color=TINTA)
ax.set_xlabel("Diferença de jogadores vivos no momento da foto", color=TINTA)
salvar(fig, "acuracia_por_vantagem.png")

# --- 3. Importância por permutação em grupo (rótulos corrigidos) ---
X_tr, X_te, y_tr, y_te = carregar_dados(corrigir_rotulos=True)
modelo = gradient_boosting().fit(X_tr, y_tr)
acc_base = accuracy_score(y_te, modelo.predict(X_te))
colunas = X_te.columns
grupos = {
    "Vida": ["ct_health", "t_health"],
    "Jogadores vivos": ["ct_players_alive", "t_players_alive"],
    "Bomba plantada": ["bomb_planted"],
    "Tempo restante": ["time_left"],
    "Armadura + capacete": ["ct_armor", "t_armor", "ct_helmets", "t_helmets"],
    "Armas": [c for c in colunas if "_weapon_" in c],
    "Granadas": [c for c in colunas if "_grenade_" in c],
    "Kits de defuse": ["ct_defuse_kits"],
    "Dinheiro": ["ct_money", "t_money"],
    "Placar": ["ct_score", "t_score"],
    "Mapa": [c for c in colunas if c.startswith("map_")],
}
rng = np.random.default_rng(42)
quedas = {}
for nome, cols in grupos.items():
    resultados = []
    for _ in range(5):
        X_embaralhado = X_te.copy()
        X_embaralhado[cols] = X_te[cols].values[rng.permutation(len(X_te))]
        resultados.append(acc_base - accuracy_score(y_te, modelo.predict(X_embaralhado)))
    quedas[nome] = np.mean(resultados) * 100
quedas = pd.Series(quedas).sort_values()
print("\nImportância (rótulos corrigidos), queda em pontos:")
print(quedas.sort_values(ascending=False).round(2).to_string())

fig, ax = novo_grafico("O que o modelo usa (importância por permutação)", (7, 4.8))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRADE, linewidth=0.8)
barras = ax.barh(quedas.index, quedas.values, color=AZUL, height=0.6)
for barra in barras:
    largura = barra.get_width()
    ax.text(max(largura, 0) + 0.15, barra.get_y() + barra.get_height() / 2,
            f"{largura:.1f}", va="center", color=TINTA, fontsize=9)
ax.axvline(0, color=TINTA_2, linewidth=0.8)
ax.set_xlabel("Pontos de acurácia perdidos", color=TINTA)
salvar(fig, "importancia_variaveis.png")

# --- 4. "E se os CTs fossem morrendo?" (modelo final da demonstração) ---
cts = [5, 4, 3, 2, 1]
probs = [probabilidade_ct(mapa="de_inferno", tempo=90, bomba=False, ct_vivos=n, t_vivos=5,
                          ct_armas={"m4a1s": n, "usps": n}, t_armas={"ak47": 5, "glock": 5}) * 100
         for n in cts]
fig, ax = novo_grafico("Chance do CT contra 5 T vivos (Inferno, 90s, rifles)")
barras = ax.bar([f"{n} CT" for n in cts], probs, color=AZUL, width=0.55)
rotular_barras(ax, barras)
ax.set_ylim(0, 100)
ax.set_ylabel("Probabilidade de vitória do CT (%)", color=TINTA)
salvar(fig, "e_se_cts_morrendo.png")
