# Manipulação de dados
import numpy as np
import pandas as pd

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Divisão dos dados
from sklearn.model_selection import train_test_split

# Algoritmos de Machine Learning
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Métricas de performance
from sklearn import metrics
from sklearn.metrics import (f1_score,
                             accuracy_score,
                             recall_score,
                             precision_score,
                             confusion_matrix,
                             roc_auc_score)

# Ajustes de Hiperparametros
from sklearn.model_selection import GridSearchCV

# Optional para Annotations das funções
from typing import Optional

# Ignorar alertas
import warnings

warnings.filterwarnings('ignore')

def prepare_dataset(data):

    print(data.head())

    # Verifica o shape dos dados
    print(f'Shape dos dados: {data.shape}\n')

    print(f'Esta base de dados tem {data.shape[0]} linhas e {data.shape[1]} colunas.')

    print(data.describe())

    # Lista de variáveis categóricas
    colunas_cat = data.select_dtypes(include=['object']).columns.tolist()

    # Loop para imprimir a contagem de valores únicos em cada coluna categórica
    for coluna in colunas_cat:
        print(f'### Coluna <{coluna}> ###')
        print(data[coluna].value_counts())
        print('-' * 40)

    print(data.info())

    # Verificando dados nulos
    print('Colunas com dados nulos:')
    print(data.isnull().sum()[data.isnull().sum() > 0])

    # Corrige o erro de digitação
    corrige_carro = {'carr0': 'carro'}
    data.replace(corrige_carro, inplace=True)

    # Verifica as categorias novamente
    print(data['motivo'].value_counts())

    # Convertendo variáveis Categóricas Ordinais
    conversao_variaveis = {
        'saldo_corrente': {
            'desconhecido': -1,
            '< 0 DM': 1,
            '1 - 200 DM': 2,
            '> 200 DM': 3,
        },
        'historico_credito': {
            'critico': 1,
            'ruim': 2,
            'bom': 3,
            'muito bom': 4,
            'perfeito': 5
        },
        'saldo_poupanca': {
            'desconhecido': -1,
            '< 100 DM': 1,
            '100 - 500 DM': 2,
            '500 - 1000 DM': 3,
            '> 1000 DM': 4,
        },
        'tempo_empregado': {
            'desempregado': 1,
            '< 1 ano': 2,
            '1 - 4 anos': 3,
            '4 - 7 anos': 4,
            '> 7 anos': 5,
        },
        'telefone': {
            'nao': 1,
            'sim': 2,
        }
    }

    data.replace(conversao_variaveis, inplace=True)

    # Gera a lista de variáveis categóricas
    cols_cat = data.select_dtypes(include='object').columns.tolist()

    # Removendo 'inadimplente' pois é nossa variável Alvo
    cols_cat.remove('inadimplente')

    print(cols_cat)

    # Implementa o OneHotEncoding
    data = pd.get_dummies(data, columns=cols_cat, drop_first=True)

    print(data.head())

    # Convertendo a variável alvo
    conversao_alvo = {
        'inadimplente': {'nao': 0, 'sim': 1}
    }

    data.replace(conversao_alvo, inplace=True)
    print(data['inadimplente'])

    # Imputando os valores nulos com a média
    data = data.fillna(data.mean())

    # Verifica valores nulos novamente
    print(data.isnull().sum())

    print(data.head())

    # Variáveis independentes (características)
    data_x = data.drop(['inadimplente'], axis=1)

    # Variável dependente (alvo)
    data_y = data['inadimplente']

    return data_x, data_y

def train_trees(dataset_input_x, dataset_input_y):

    # Divisão dos dados em Treino e Teste
    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = train_test_split(dataset_input_x,
                                                        dataset_input_y,
                                                        test_size=0.30,
                                                        random_state=1,
                                                        stratify=dataset_input_y)  # mantém as proporções das classes

    # Instanciando o Modelo
    # arvore_d = DecisionTreeClassifier(random_state=1, max_depth=7, max_leaf_nodes=10, min_impurity_decrease=0.001)
    # arvore_d = RandomForestClassifier(n_estimators=100, max_depth=9)
    arvore_d = ExtraTreesClassifier(n_estimators=200, max_depth=11)

    # Resultados dataset teste DecisionTreeClassifier:
    # Acurácia na base de teste: 72.0%
    # Precisão na base de teste: 55.00000000000001%
    # Recall na base de teste: 47.0%
    # F1 score na base de teste: 50.0%

    # Resultados dataset teste RandomForestClassifier:
    # Acurácia na base de teste: 76.0%
    # Precisão na base de teste: 69.0%
    # Recall na base de teste: 38.0%
    # F1 score na base de teste: 49.0%

    # Resultados dataset teste ExtraTreesClassifier:
    # Acurácia na base de teste: 78.0%
    # Precisão na base de teste: 77.0%
    # Recall na base de teste: 37.0%
    # F1 score na base de teste: 50.0%

    # Treinando o modelo
    arvore_d.fit(dataset_x_train, dataset_y_train)

    dataset_y_test_pred = arvore_d.predict(dataset_x_test)
    dataset_y_train_pred = arvore_d.predict(dataset_x_train)

    cm_train = metrics.confusion_matrix(dataset_y_train, dataset_y_train_pred)
    acc_train = metrics.accuracy_score(dataset_y_train, dataset_y_train_pred)
    prec_train = metrics.precision_score(dataset_y_train, dataset_y_train_pred)
    rec_train = metrics.recall_score(dataset_y_train, dataset_y_train_pred)
    f1_score_train = metrics.f1_score(dataset_y_train, dataset_y_train_pred)

    cm_test = metrics.confusion_matrix(dataset_y_test, dataset_y_test_pred)
    acc_test = metrics.accuracy_score(dataset_y_test, dataset_y_test_pred)
    prec_test = metrics.precision_score(dataset_y_test, dataset_y_test_pred)
    rec_test = metrics.recall_score(dataset_y_test, dataset_y_test_pred)
    f1_score_test = metrics.f1_score(dataset_y_test, dataset_y_test_pred)

    print('Resultados dataset treino:')
    print(f'Acurácia na base de treino: {round(acc_train, 2) * 100}%')
    print(f'Precisão na base de treino: {round(prec_train, 2) * 100}%')
    print(f'Recall na base de treino: {round(rec_train, 2) * 100}%')
    print(f'F1 score na base de treino: {round(f1_score_train, 2) * 100}%')

    # Plot da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_train, annot=True, fmt='g')
    plt.title('Matriz de Confusão: Base de Treino', weight='bold')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    plt.show()

    print('\nResultados dataset teste:')
    print(f'Acurácia na base de teste: {round(acc_test, 2) * 100}%')
    print(f'Precisão na base de teste: {round(prec_test, 2) * 100}%')
    print(f'Recall na base de teste: {round(rec_test, 2) * 100}%')
    print(f'F1 score na base de teste: {round(f1_score_test, 2) * 100}%')
    print(metrics.classification_report(dataset_y_test, dataset_y_test_pred))

    # Plot da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_test, annot=True, fmt='g')
    plt.title('Matriz de Confusão: Base de Teste', weight='bold')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    plt.show()

    # feature_names = list(dataset_x_train.columns)
    # plt.figure(figsize=(20, 30))
    # tree.plot_tree(arvore_d, feature_names=feature_names, filled=True,
    #                fontsize=9, node_ids=True, class_names=True)
    # plt.show()


if __name__ == '__main__':

    dataset_complete = pd.read_csv('credito.csv')
    dataset_x, dataset_y = prepare_dataset(dataset_complete)

    train_trees(dataset_x, dataset_y)


