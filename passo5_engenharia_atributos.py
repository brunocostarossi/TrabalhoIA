from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from preparacao import carregar_dados

# Validação cruzada temporal: divide o TREINO em 5 blocos em ordem e
# sempre treina no passado e valida no bloco seguinte. O teste fica guardado.
validacao = TimeSeriesSplit(n_splits=5)

versoes = {
    "original": dict(),
    "sem placar": dict(sem_placar=True),
    "novos atributos": dict(novos_atributos=True),
    "novos atributos, sem placar": dict(novos_atributos=True, sem_placar=True),
}

modelos = {
    "Logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=42),
}

print(f"{'versão dos dados':30s}" + "".join(f"{m:>20s}" for m in modelos))
for nome_versao, opcoes in versoes.items():
    X_treino, _, y_treino, _ = carregar_dados(**opcoes)
    linha = f"{nome_versao:30s}"
    for criar_modelo in modelos.values():
        notas = cross_val_score(criar_modelo(), X_treino, y_treino, cv=validacao)
        linha += f"{notas.mean():>14.4f} ±{notas.std():.3f}"
    print(linha)
