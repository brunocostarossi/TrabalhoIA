import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from preparacao import carregar_dados

X_treino, _, y_treino, _ = carregar_dados(novos_atributos=True, corrigir_rotulos=True)
validacao = TimeSeriesSplit(n_splits=5)


def rede(camadas):
    # Redes neurais também precisam das colunas na mesma escala.
    # early_stopping: separa 10% do treino e para quando a nota nele parar de subir.
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=camadas, early_stopping=True, max_iter=300, random_state=42))


modelos = {
    "Gradient Boosting (referência)": HistGradientBoostingClassifier(random_state=42),
    "Rede: 1 camada (64)": rede((64,)),
    "Rede: 2 camadas (128, 64)": rede((128, 64)),
    "Rede: 3 camadas (256, 128, 64)": rede((256, 128, 64)),
}

for nome, modelo in modelos.items():
    inicio = time.time()
    notas = cross_val_score(modelo, X_treino, y_treino, cv=validacao)
    print(f"{nome:32s} validação: {notas.mean():.4f} ±{notas.std():.3f}  ({time.time() - inicio:.0f}s)")
