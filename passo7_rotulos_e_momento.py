import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score

from preparacao import carregar_dados, limpar_rotulos, marcar_rounds

# --- 1. Quem vence mais: por FOTO ou por ROUND? ---
df = limpar_rotulos(marcar_rounds(pd.read_csv("csgo_round_snapshots.csv")))
rounds = df.groupby("round_id").agg(vencedor=("round_winner", "first"),
                                    fotos=("round_winner", "size"))
print("Vitórias por FOTO: ", df["round_winner"].value_counts(normalize=True).round(3).to_dict())
print("Vitórias por ROUND:", rounds["vencedor"].value_counts(normalize=True).round(3).to_dict())
print("Fotos por round:   ", rounds.groupby("vencedor")["fotos"].mean().round(2).to_dict())
print()

# --- 2. Comparar versões com validação cruzada temporal (só no treino) ---
validacao = TimeSeriesSplit(n_splits=5)
versoes = {
    "rótulos corrigidos": dict(corrigir_rotulos=True),
    "+ novos atributos": dict(corrigir_rotulos=True, novos_atributos=True),
    "+ novos + momento": dict(corrigir_rotulos=True, novos_atributos=True, atributos_momento=True),
}
for nome, opcoes in versoes.items():
    X_treino, _, y_treino, _ = carregar_dados(**opcoes)
    notas = cross_val_score(HistGradientBoostingClassifier(random_state=42),
                            X_treino, y_treino, cv=validacao)
    print(f"{nome:22s} validação: {notas.mean():.4f} ±{notas.std():.3f}")

# --- 3. Teste final, uma vez só, com a melhor versão ---
X_treino, X_teste, y_treino, y_teste = carregar_dados(**versoes["+ novos + momento"])
modelo = HistGradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
acertos = modelo.predict(X_teste) == y_teste
print(f"\nTeste final: {acertos.mean():.4f}")
print(f"  período de compra: {acertos[X_teste['time_left'] > 170].mean():.4f}")
print(f"  round em andamento: {acertos[X_teste['time_left'] <= 170].mean():.4f}")
