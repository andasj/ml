import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib


def hello_world_linear_regression():
    # Load the diabetes dataset
    diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

    print(diabetes_y)

    # Use only one feature
    diabetes_x = diabetes_x[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_x_train = diabetes_x[:-20]
    diabetes_x_test = diabetes_x[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_x_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,
                                                          diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test,
                                                          diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_x_test, diabetes_y_test, color="black")
    plt.plot(diabetes_x_test, diabetes_y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())

    plt.show()


def save_csv_dataset_diabetes(name):

    diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
    features_names = datasets.load_diabetes().feature_names
    df_x = pd.DataFrame(diabetes_x, columns=features_names)
    df_y = pd.DataFrame(diabetes_y, columns=["target"])
    result = pd.concat([df_x, df_y], axis=1)
    result.to_csv(name, index=False)


def open_dataset_diabetes(type_dataset='csv'):

    if type_dataset == 'csv':
        df = pd.read_csv('dataset_diabetes.csv')
        data_diabetes_x = df.iloc[:, :-1]
        data_diabetes_y = df.iloc[:, -1]
        data_diabetes_y = data_diabetes_y.to_list()

    elif type_dataset == 'scikit':
        data_diabetes = datasets.load_diabetes()
        df = pd.DataFrame(data=data_diabetes.data, columns=data_diabetes.feature_names)
        data_diabetes_x = df
        data_diabetes_y = data_diabetes.target

    else:
        print(f'{type_dataset} não identificado.')
        data_diabetes_x, data_diabetes_y = datasets.load_diabetes(return_X_y=True)

    return data_diabetes_x, data_diabetes_y

def show_dataset(dataset_x, dataset_y):

    features_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    df_x = dataset_x
    df_y = pd.DataFrame(dataset_y, columns=["target"])
    print('\nShow info dataset')
    print(df_x.describe())
    print(df_y.describe())
    print(df_x.info())
    print('\n')

    sns.pairplot(df_x.join(df_y))
    plt.show()

    correlation_matrix = df_x.join(df_y).corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    for i in range(10):
        dataset_filter_x = dataset_x[dataset_x.columns[i]].to_list()
        plt.subplot(5, 2, i+1)
        plt.scatter(dataset_filter_x, dataset_y)
        plt.xlabel(features_names[i])
        plt.ylabel('target')
        plt.xticks([]), plt.yticks([])

    plt.show()


def train_linear_regression(dataset_x, dataset_y, name_model_train):

    # percent_train = 0.8
    # size_train = int(percent_train * len(dataset_x))
    # size_test = len(dataset_x) - size_train
    # dataset_x_train, dataset_x_test = (dataset_x[:-size_test],
    #                                    dataset_x[-size_test:])
    # dataset_y_train, dataset_y_test = (data_input_y[:-size_test],
    #                                    data_input_y[-size_test:])
    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = (
        train_test_split(dataset_x, dataset_y,
                         test_size=0.2, random_state=4))

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # regr = linear_model.Ridge()
    # regr = linear_model.Lasso()
    # regr = linear_model.ElasticNet()
    # regr = linear_model.BayesianRidge()
    # regr = linear_model.SGDRegressor()

    # Train the model using the training sets
    regr.fit(dataset_x_train, dataset_y_train)

    # Make predictions using the testing set
    dataset_y_pred = regr.predict(dataset_x_test)

    # Show metrics result
    print('\nShow metrics result train')
    print(f'Coeficientes: a -> {regr.coef_}, b -> {regr.intercept_}')
    print("MSE: %.2f" % mean_squared_error(dataset_y_test, dataset_y_pred))
    print("R2_score: %.2f" % r2_score(dataset_y_test, dataset_y_pred))
    print('\n')

    # Show results
    result_train = regr.predict(dataset_x_train)
    result_test = dataset_y_pred
    result_total = regr.predict(dataset_x)

    len_dataset_x = len(dataset_x.columns)

    for i in range(len_dataset_x):

        plt.subplot(3, 1, 1)
        plt.scatter(dataset_x_train[dataset_x_train.columns[i]].to_list(), dataset_y_train, color="black")
        plt.scatter(dataset_x_train[dataset_x_train.columns[i]].to_list(), result_train, color="blue")
        plt.ylabel('train')
        plt.title(dataset_x_train.columns[i])
        plt.xticks([])

        plt.subplot(3, 1, 2)
        plt.scatter(dataset_x_test[dataset_x_train.columns[i]].to_list(), dataset_y_test, color="black")
        plt.scatter(dataset_x_test[dataset_x_train.columns[i]].to_list(), result_test, color="blue")
        plt.ylabel('test')
        plt.title(dataset_x_train.columns[i])
        plt.xticks([])

        plt.subplot(3, 1, 3)
        plt.scatter(dataset_x[dataset_x.columns[i]].to_list(), dataset_y, color="black")
        plt.scatter(dataset_x[dataset_x.columns[i]].to_list(), result_total, color="blue")
        plt.ylabel('total')
        plt.title(dataset_x_train.columns[i])

        plt.tight_layout()
        plt.show()
        plt.clf()

    joblib.dump(regr, name_model_train)


def predict_linear_regression(name_model_linear_regression, x_input):

    # Carregando o modelo do arquivo
    regr = joblib.load(name_model_linear_regression)

    # Fazendo previsões com o modelo carregado
    y_pred = regr.predict([x_input])

    return y_pred[0]


if __name__ == '__main__':

    name_model = 'diabetes_model_linear_regression.pkl'

    # hello_world_linear_regression()
    # save_csv_dataset_diabetes('dataset_diabetes.csv')

    data_complete_x, data_y = open_dataset_diabetes('scikit')
    data_x = data_complete_x[['bmi', 's5']]
    # show_dataset(data_complete_x, data_y)
    train_linear_regression(data_x, data_y, name_model)
    
    # result_pred = predict_linear_regression(name_model, data_x.iloc[0].tolist())
    # print(f'\nPredict\nInput: {data_x.iloc[0].tolist()} - Predict result: {result_pred}')
    # -----------------------------------------------------------------------------






