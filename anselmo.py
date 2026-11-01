import pandas as pd

df = pd.read_csv("csgo_round_snapshots.csv")

print("Formato (linhas, colunas):", df.shape)
print()
print(df.head())
print()
print(df.dtypes.value_counts())
print()
print("Distribuição do alvo:")
print(df["round_winner"].value_counts(normalize=True))
print()
print("Valores faltantes:", df.isna().sum().sum())