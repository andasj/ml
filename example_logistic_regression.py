import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib

def open_dataset_diabetes():

    df = pd.read_csv('pima-indians-diabetes.csv')
    data_diabetes_x = df.iloc[:, :-1]
    data_diabetes_y = df.iloc[:, -1]
    data_diabetes_y = data_diabetes_y.to_list()

    return data_diabetes_x, data_diabetes_y

def show_dataset(dataset_x, dataset_y):

    df_x = dataset_x
    # features_names = df_x.columns.tolist()
    df_y = pd.DataFrame(dataset_y, columns=["Class"])
    print('\nShow info dataset')
    print(f'\ndf_x.describe(): \n{df_x.describe()}')
    print(f'\ndf_y.describe(): \n{df_y.describe()}')
    info_class = df_x['Pregnancies'].value_counts()
    print(f'Classe Pregnancies: {info_class}')
    print(f'\ndf_x.info(): \n')
    dataset_x.info()
    print('\n\n')

    sns.set_style('darkgrid')
    df_x.join(df_y).hist(figsize=(15, 10))
    plt.show()

    # sns.pairplot(df_x.join(df_y))
    # plt.show()
    #
    # correlation_matrix = df_x.join(df_y).corr()
    # sns.heatmap(correlation_matrix, annot=True)
    # plt.xticks(rotation=30)
    # plt.show()

def prepare_dataset(change_dataset):

    cols = ['Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'Pedigree']

    change_dataset[cols] = change_dataset[cols].replace(0, np.nan)
    change_dataset[cols] = change_dataset[cols].fillna(change_dataset[cols].mean())

    # print(change_dataset.isnull().sum())

    return change_dataset

def train_logistic_regression(data_x, data_y, name_model_train):

    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = (
        train_test_split(data_x, data_y, test_size=0.2, random_state=4))

    regr = linear_model.LogisticRegression(solver='lbfgs', penalty='l2', max_iter=200)
    # regr = linear_model.LogisticRegression(solver='liblinear', max_iter=200)
    # regr = linear_model.LogisticRegression(solver='newton-cg', max_iter=200)
    # regr = linear_model.LogisticRegression(solver='newton-cholesky', max_iter=200)
    # regr = linear_model.LogisticRegression(solver='sag', max_iter=200)
    # regr = linear_model.LogisticRegression(solver='saga', max_iter=200)

    regr.fit(dataset_x_train, dataset_y_train)

    dataset_y_test_pred = regr.predict(dataset_x_test)
    dataset_y_train_pred = regr.predict(dataset_x_train)

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

    # # Plot da Matriz de Confusão
    # plt.figure(figsize=(7, 5))
    # sns.heatmap(cm_train, annot=True, fmt='g')
    # plt.title('Matriz de Confusão: Base de Treino', weight='bold')
    # plt.xlabel('Valores Previstos')
    # plt.ylabel('Valores Reais')
    # plt.show()

    print('\nResultados dataset teste:')
    print(f'Acurácia na base de teste: {round(acc_test, 2) * 100}%')
    print(f'Precisão na base de teste: {round(prec_test, 2) * 100}%')
    print(f'Recall na base de teste: {round(rec_test, 2) * 100}%')
    print(f'F1 score na base de teste: {round(f1_score_test, 2) * 100}%')

    # Plot da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_test, annot=True, fmt='g')
    plt.title('Matriz de Confusão: Base de Teste', weight='bold')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    plt.show()

    joblib.dump(regr, name_model_train)


def predict_linear_regression(name_model_logistic_regression, x_input):

    # Carregando o modelo do arquivo
    regr = joblib.load(name_model_logistic_regression)

    # Fazendo previsões com o modelo carregado
    y_pred = regr.predict(x_input)

    return y_pred[0]


if __name__ == '__main__':

    name_model = 'diabetes_model_logistic_regression.pkl'

    data_complete_x, data_y = open_dataset_diabetes()
    data_complete_x = prepare_dataset(data_complete_x)

    # show_dataset(data_complete_x, data_y)
    train_logistic_regression(data_complete_x, data_y, name_model)

    # number = random.randint(0, 768)
    # result_pred = predict_linear_regression(name_model, data_complete_x.iloc[[number]])
    # print(f'\nPredict\nInput\n: {data_complete_x.iloc[[number]]} '
    #       f'\nPredict result: {result_pred} \nReal result: {data_y[number]}')


