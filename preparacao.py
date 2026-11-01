import pandas as pd
from sklearn.model_selection import train_test_split

# Preços aproximados do CS:GO (variaram entre atualizações do jogo).
# Para o modelo, o que importa é a ordem de grandeza: AWP >> AK > pistola.
PRECOS = {
    "ak47": 2700, "aug": 3300, "awp": 4750, "bizon": 1400, "cz75auto": 500,
    "elite": 400, "famas": 2050, "g3sg1": 5000, "galilar": 1800, "glock": 200,
    "m249": 5200, "m4a1s": 2900, "m4a4": 3100, "mac10": 1050, "mag7": 1300,
    "mp5sd": 1500, "mp7": 1500, "mp9": 1250, "negev": 1700, "nova": 1050,
    "p90": 2350, "r8revolver": 600, "sawedoff": 1100, "scar20": 5000,
    "sg553": 3000, "ssg08": 1700, "ump45": 1200, "xm1014": 2000,
    "deagle": 700, "fiveseven": 500, "usps": 200, "p250": 300, "p2000": 200,
    "tec9": 500,
    "hegrenade": 300, "flashbang": 200, "smokegrenade": 300,
    "incendiarygrenade": 600, "molotovgrenade": 400, "decoygrenade": 50,
}
PRECO_COLETE, PRECO_CAPACETE, PRECO_KIT = 650, 350, 400


def valor_equipamento(X, lado):
    """Soma o valor em dinheiro de tudo que o time 'ct' ou 't' carrega."""
    total = 0
    for item, preco in PRECOS.items():
        for tipo in ("weapon", "grenade"):
            coluna = f"{lado}_{tipo}_{item}"
            if coluna in X:
                total = total + X[coluna] * preco
    # ct_armor é a soma dos pontos de colete (0 a 100 por jogador)
    total = total + X[f"{lado}_armor"] / 100 * PRECO_COLETE
    total = total + X[f"{lado}_helmets"] * PRECO_CAPACETE
    if lado == "ct":
        total = total + X["ct_defuse_kits"] * PRECO_KIT
    return total


def criar_atributos(X):
    """Cria colunas que resumem o estado do round como 'vantagem do CT'."""
    X = X.copy()
    X["ct_equip_valor"] = valor_equipamento(X, "ct")
    X["t_equip_valor"] = valor_equipamento(X, "t")

    # Diferenças CT - T: positivo = vantagem do CT
    X["dif_jogadores"] = X["ct_players_alive"] - X["t_players_alive"]
    X["dif_vida"] = X["ct_health"] - X["t_health"]
    X["dif_armadura"] = X["ct_armor"] - X["t_armor"]
    X["dif_dinheiro"] = X["ct_money"] - X["t_money"]
    X["dif_equip_valor"] = X["ct_equip_valor"] - X["t_equip_valor"]

    # Bomba plantada com pouco tempo sobrando é ruim para o CT
    X["bomba_x_tempo"] = X["bomb_planted"] * X["time_left"]
    return X


def primeira_foto_da_compra(df):
    """Marca a 1ª foto do período de compra (time_left > 170) após cada round."""
    compra = df["time_left"] > 170
    return compra & ~compra.shift(fill_value=False)


def marcar_rounds(df):
    """Numera os rounds: um round novo começa no período de compra.

    Precisa rodar ANTES de limpar_rotulos, que remove justamente a foto
    que marca o início de muitos rounds.
    """
    df = df.copy()
    df["round_id"] = primeira_foto_da_compra(df).cumsum()
    return df


def limpar_rotulos(df):
    """Remove a 1ª foto do período de compra após cada round.

    Investigando o CSV, essa foto já mostra o round novo, mas o rótulo é o
    vencedor do round que acabou de terminar (99,6% dos casos). As fotos
    seguintes do período de compra já têm o rótulo do round novo.
    """
    return df[~primeira_foto_da_compra(df)].reset_index(drop=True)


def criar_atributos_momento(X, round_id):
    """O que mudou desde a foto anterior do MESMO round (só usa o passado)."""
    X = X.copy()
    for coluna in ["ct_players_alive", "t_players_alive", "ct_health", "t_health"]:
        X[f"var_{coluna}"] = X.groupby(round_id)[coluna].diff().fillna(0)
    X["var_dif_jogadores"] = X["var_ct_players_alive"] - X["var_t_players_alive"]
    return X


def carregar_dados(caminho="csgo_round_snapshots.csv", novos_atributos=False, sem_placar=False,
                   corrigir_rotulos=False, atributos_momento=False):
    """Lê o CSV, transforma tudo em número e separa treino/teste (em ordem)."""
    df = marcar_rounds(pd.read_csv(caminho))
    if corrigir_rotulos:
        df = limpar_rotulos(df)
    round_id = df.pop("round_id")

    y = (df["round_winner"] == "CT").astype(int)  # CT = 1, T = 0
    X = df.drop(columns=["round_winner"])
    X["bomb_planted"] = X["bomb_planted"].astype(int)
    X = pd.get_dummies(X, columns=["map"], dtype=int)

    if novos_atributos:
        X = criar_atributos(X)
    if atributos_momento:
        X = criar_atributos_momento(X, round_id)
    if sem_placar:
        X = X.drop(columns=["ct_score", "t_score"])

    # shuffle=False para evitar vazamento entre fotos do mesmo round
    return train_test_split(X, y, test_size=0.2, shuffle=False)
