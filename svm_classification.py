from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import random

def prepare_dataset():

    cancer = datasets.load_breast_cancer()
    # print("Features: ", cancer.feature_names)
    # print("Labels: ", cancer.target_names)
    # print("Shape: ", cancer.data.shape)
    # print(cancer.target)
    # print(cancer.data[0:5])

    return cancer.data, cancer.target

def show_dataset():

    cancer = datasets.load_breast_cancer()

    df_x = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df_y = pd.DataFrame(cancer.target, columns=["Class"])

    print('\nShow info dataset')
    print(f'\ndf_x.describe(): \n{df_x.describe()}')
    print(f'\ndf_y.describe(): \n{df_y.describe()}')
    print(f'\ndf_x.info(): \n')
    df_x.info()
    print('\n\n')

    sns.set_style('darkgrid')
    df_x.join(df_y).hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    correlation_matrix = df_x.join(df_y).corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.xticks(rotation=30)
    plt.show()

def train_svm(data_x, data_y, name_model_train):

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3,random_state=109) # 70% training and 30% test
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(x_train, y_train)

    y_test_pred = clf.predict(x_test)
    # y_train_pred = clf.predict(x_train)

    # cm_train = metrics.confusion_matrix(y_train, y_train_pred)
    # acc_train = metrics.accuracy_score(y_train, y_train_pred)
    # prec_train = metrics.precision_score(y_train, y_train_pred)
    # rec_train = metrics.recall_score(y_train, y_train_pred)
    # f1_score_train = metrics.f1_score(y_train, y_train_pred)

    cm_test = metrics.confusion_matrix(y_test, y_test_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    prec_test = metrics.precision_score(y_test, y_test_pred)
    rec_test = metrics.recall_score(y_test, y_test_pred)
    f1_score_test = metrics.f1_score(y_test, y_test_pred)

    # print('Resultados dataset treino:')
    # print(f'Acurácia na base de treino: {round(acc_train, 2) * 100}%')
    # print(f'Precisão na base de treino: {round(prec_train, 2) * 100}%')
    # print(f'Recall na base de treino: {round(rec_train, 2) * 100}%')
    # print(f'F1 score na base de treino: {round(f1_score_train, 2) * 100}%')
    #
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
    print(metrics.classification_report(y_test, y_test_pred))

    # Plot da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_test, annot=True, fmt='g')
    plt.title('Matriz de Confusão: Base de Teste', weight='bold')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    plt.show()

    joblib.dump(clf, name_model_train)

def predict_svm(name_model_svm, x_input):

    # Carregando o modelo do arquivo
    model_svm = joblib.load(name_model_svm)

    # Fazendo previsões com o modelo carregado
    y_pred = model_svm.predict(x_input)

    return y_pred[0]

if __name__ == '__main__':

    name_model = 'models/cancer_svm.pkl'
    x, y = prepare_dataset()
    show_dataset()

    train_svm(x, y, name_model)

    number = random.randint(0, 569)
    result_pred = predict_svm(name_model, [x[number]])
    print(f'\nPredict\nInput:\n {[x[number]]}'
          f'\nPredict result: {result_pred} \nReal result: {y[number]}')
