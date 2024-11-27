import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib
import random

def prepare_dataset(filename_dataset):

    df = pd.read_csv(filename_dataset)
    print(df.shape)
    # print(df.head())
    # print(df.info())

    # sns.countplot(data=df,x='purpose',hue='not.fully.paid')
    # plt.xticks(rotation=45, ha='right')
    # plt.show()

    pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)

    # print(pre_df.info())

    df_x = pre_df.drop('not.fully.paid', axis=1)
    df_y = pre_df['not.fully.paid']

    return df_x, df_y

def show_dataset(df_x, df_y):

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

def train_bayes(data_x, data_y, name_model_train):

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=125)

    model = GaussianNB()

    model.fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    # y_train_pred = model.predict(x_train)

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

    joblib.dump(model, name_model_train)

def predict_bayes(name_model_svm, x_input):

    # Carregando o modelo do arquivo
    model_bayes = joblib.load(name_model_svm)

    # Fazendo previsões com o modelo carregado
    y_pred = model_bayes.predict(x_input)

    return y_pred[0]

if __name__ == '__main__':

    name_model = 'models/loan_data_bayes.pkl'
    x, y = prepare_dataset('loan_data.csv')
    show_dataset(x, y)

    train_bayes(x, y, name_model)

    number = random.randint(0, 9578)
    result_pred = predict_bayes(name_model, x.iloc[[number]])
    print(f'\nPredict\nInput\n: {x.iloc[[number]]} '
          f'\nPredict result: {result_pred} \nReal result: {y[number]}')

