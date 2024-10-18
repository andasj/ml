import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Algoritmos de Machine Learning
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def prepare_dataset_credito(data):

    # print(f'Esta base de dados tem {data.shape[0]} linhas e {data.shape[1]} colunas.')
    # print(data.describe())
    # print(data.info())

    # # Loop para imprimir a contagem de valores únicos em cada coluna categórica
    # for coluna in data.select_dtypes(include=['object']).columns.tolist():
    #     print(f'\n### Coluna <{coluna}> ###')
    #     print(data[coluna].value_counts())
    #     print('-' * 40)

    # Corrige o erro de digitação
    corrige_carro = {'carr0': 'carro'}
    data.replace(corrige_carro, inplace=True)

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
    cols_cat.remove('inadimplente')
    data = pd.get_dummies(data, columns=cols_cat, drop_first=True)

    # Convertendo a variável alvo
    conversao_alvo = {'inadimplente': {'nao': 0, 'sim': 1}}
    data.replace(conversao_alvo, inplace=True)

    # Imputando os valores nulos com a média
    data = data.fillna(data.mean())

    data_x = data.drop(['inadimplente'], axis=1)
    data_y = data['inadimplente']

    # print('\nDescribe dataset x:\n',data_x.describe())
    # print(data_x.info())
    # print('\nDescribe dataset y:\n',data_y.describe())

    return data_x, data_y

def train_trees(dataset_input_x, dataset_input_y, name_model_train):

    # Divisão dos dados em Treino e Teste
    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = train_test_split(dataset_input_x,
                                                        dataset_input_y,
                                                        test_size=0.30,
                                                        random_state=1,
                                                        stratify=dataset_input_y)  # mantém as proporções das classes

    # Instanciando o Modelo
    arvore_d = DecisionTreeClassifier(random_state=1, max_depth=7, max_leaf_nodes=10,
                                      min_impurity_decrease=0.001, criterion="log_loss", splitter="best")
    # arvore_d = RandomForestClassifier(n_estimators=100, max_depth=9)
    # arvore_d = ExtraTreesClassifier(n_estimators=200, max_depth=11)

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
    print(f'Matriz de confusão de treino:\n {cm_train}')
    print(f'Acurácia na base de treino: {round(acc_train, 2) * 100}%')
    print(f'Precisão na base de treino: {round(prec_train, 2) * 100}%')
    print(f'Recall na base de treino: {round(rec_train, 2) * 100}%')
    print(f'F1 score na base de treino: {round(f1_score_train, 2) * 100}%')

    # # Plot da Matriz de Confusão
    # plt.figure(figsize=(7, 5))
    # sns.heatmap(cm_train, annot=True, fmt='g')
    # plt.title('Matriz de Confusão: Base de Treino', weight='bold')
    # plt.xlabel('Valores Previstos')
    # plt.ylabel('Valores Reais')
    # plt.show()

    print('\nResultados dataset teste:')
    print(f'Matriz de confusão de teste:\n {cm_test}')
    print(f'Acurácia na base de teste: {round(acc_test, 2) * 100}%')
    print(f'Precisão na base de teste: {round(prec_test, 2) * 100}%')
    print(f'Recall na base de teste: {round(rec_test, 2) * 100}%')
    print(f'F1 score na base de teste: {round(f1_score_test, 2) * 100}%')

    # print(metrics.classification_report(dataset_y_test, dataset_y_test_pred))

    # # Plot da Matriz de Confusão
    # plt.figure(figsize=(7, 5))
    # sns.heatmap(cm_test, annot=True, fmt='g')
    # plt.title('Matriz de Confusão: Base de Teste', weight='bold')
    # plt.xlabel('Valores Previstos')
    # plt.ylabel('Valores Reais')
    # plt.show()

    # feature_names = list(dataset_x_train.columns)
    # plt.figure(figsize=(20, 30))
    # tree.plot_tree(arvore_d, feature_names=feature_names, filled=True,
    #                fontsize=9, node_ids=True, class_names=True)
    # plt.show()

    joblib.dump(arvore_d, name_model_train)

def predict_decision_trees(name_model_decision_trees, data_x, data_y):

    number = random.randint(0, len(data_x))

    # Carregando o modelo do arquivo
    arvore_d = joblib.load(name_model_decision_trees)

    # Fazendo previsões com o modelo carregado
    y_pred = arvore_d.predict(data_x.iloc[[number]].values)

    print(f'\nPredict\nInput: {dataset_x.iloc[[number]].values} '
          f'\nPredict result: {y_pred[0]} \nReal result: {data_y[number]}')

    return y_pred[0]

if __name__ == '__main__':

    name_model = 'credito_model_decision_trees.pkl'

    dataset_complete = pd.read_csv('credito.csv')
    dataset_x, dataset_y = prepare_dataset_credito(dataset_complete)

    # train_trees(dataset_x, dataset_y, name_model)

    result_pred = predict_decision_trees(name_model, dataset_x, dataset_y)
