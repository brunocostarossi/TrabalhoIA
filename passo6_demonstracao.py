import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from preparacao import carregar_dados, criar_atributos

# --- 1. Modelo final: o melhor que encontramos (Gradient Boosting + novos atributos)
# Agora que já avaliamos no teste, treinamos com TODOS os dados: mais exemplos,
# modelo um pouco melhor para usar "de verdade".
X_treino, X_teste, y_treino, y_teste = carregar_dados(novos_atributos=True, corrigir_rotulos=True)
X_tudo = pd.concat([X_treino, X_teste])
y_tudo = pd.concat([y_treino, y_teste])
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_tudo, y_tudo)
COLUNAS = X_tudo.columns
COLUNAS_ORIGINAIS = carregar_dados()[0].columns  # as 103 colunas antes da engenharia


def montar_linha(mapa, tempo, bomba, ct_vivos, t_vivos, ct_armas, t_armas,
                 ct_vida=None, t_vida=None, ct_dinheiro=0, t_dinheiro=0,
                 ct_colete=True, t_colete=True, ct_kits=0, ct_score=0, t_score=0):
    """Transforma uma descrição simples do round em uma linha igual às do CSV."""
    linha = pd.DataFrame(0, index=[0], columns=COLUNAS_ORIGINAIS, dtype=float)
    linha["time_left"] = tempo
    linha["bomb_planted"] = int(bomba)
    linha["ct_players_alive"], linha["t_players_alive"] = ct_vivos, t_vivos
    # Se a vida não for informada, considera todos com vida cheia
    linha["ct_health"] = ct_vida if ct_vida is not None else 100 * ct_vivos
    linha["t_health"] = t_vida if t_vida is not None else 100 * t_vivos
    linha["ct_armor"] = 100 * ct_vivos if ct_colete else 0
    linha["t_armor"] = 100 * t_vivos if t_colete else 0
    linha["ct_helmets"] = ct_vivos if ct_colete else 0
    linha["t_helmets"] = t_vivos if t_colete else 0
    linha["ct_defuse_kits"] = ct_kits
    linha["ct_money"], linha["t_money"] = ct_dinheiro, t_dinheiro
    linha["ct_score"], linha["t_score"] = ct_score, t_score
    linha[f"map_{mapa}"] = 1
    for arma, qtd in ct_armas.items():
        linha[f"ct_weapon_{arma}"] = qtd
    for arma, qtd in t_armas.items():
        linha[f"t_weapon_{arma}"] = qtd
    # Calcula as colunas da engenharia de atributos, igual no treino
    return criar_atributos(linha)[COLUNAS]


def prever(descricao, **cenario):
    prob_ct = modelo.predict_proba(montar_linha(**cenario))[0, 1]
    print(f"{descricao:55s} CT {prob_ct:6.1%} | T {1 - prob_ct:6.1%}")


# --- 2. Cenários de exemplo ---
print("=== Cenários ===")
prever("Início do round, 5x5, ambos de rifle (Mirage)",
       mapa="de_mirage", tempo=115, bomba=False, ct_vivos=5, t_vivos=5,
       ct_armas={"m4a4": 4, "awp": 1, "usps": 5}, t_armas={"ak47": 5, "glock": 5}, ct_kits=3)

prever("5x5, CT em eco (só pistola, sem colete)",
       mapa="de_mirage", tempo=115, bomba=False, ct_vivos=5, t_vivos=5,
       ct_armas={"usps": 5}, t_armas={"ak47": 5, "glock": 5}, ct_colete=False)

prever("Bomba plantada, 3 CT x 2 T, 30s restantes",
       mapa="de_mirage", tempo=30, bomba=True, ct_vivos=3, t_vivos=2,
       ct_armas={"m4a4": 3, "usps": 3}, t_armas={"ak47": 2, "glock": 2}, ct_kits=2)

prever("Bomba plantada, 1 CT x 2 T, 15s, CT sem kit",
       mapa="de_mirage", tempo=15, bomba=True, ct_vivos=1, t_vivos=2,
       ct_armas={"m4a4": 1, "usps": 1}, t_armas={"ak47": 2, "glock": 2})

# --- 3. "E se...?": muda uma coisa só e vê a probabilidade mudar ---
print("\n=== E se os CTs fossem morrendo? (5 T vivos, rifles, Inferno) ===")
for ct_vivos in range(5, 0, -1):
    prever(f"{ct_vivos} CT x 5 T",
           mapa="de_inferno", tempo=90, bomba=False, ct_vivos=ct_vivos, t_vivos=5,
           ct_armas={"m4a1s": ct_vivos, "usps": ct_vivos}, t_armas={"ak47": 5, "glock": 5})
