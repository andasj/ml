import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import viewer as vi
from sklearn import metrics
import random
import warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
# print(tf.__version__)

def prepare_dataset_mnist():

    # Carregar o dataset MNIST
    mnist = tf.keras.datasets.mnist

    # Dividir os dados em conjuntos de treinamento e teste
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    # Normalizar os dados para o intervalo [0, 1]
    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

    # Dividir o conjunto de treinamento em treino e validação
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1667, random_state=42
    )

    # Adicionar uma dimensão de canal (necessário para a camada convolucional)
    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    dict_dataset = {"x_train": x_train,
                    "x_val": x_val,
                    "x_test": x_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test}

    plt.figure(figsize=(10, 10))
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()

    return dict_dataset

def model_evaluate(model, dataset, history):

    y_pred_probs = model.predict(dataset['x_test'])
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = metrics.confusion_matrix(dataset['y_test'], y_pred)
    acc = metrics.accuracy_score(dataset['y_test'], y_pred)
    prec = metrics.precision_score(dataset['y_test'], y_pred, average='macro')
    rec = metrics.recall_score(dataset['y_test'], y_pred, average='macro')
    f1_score = metrics.f1_score(dataset['y_test'], y_pred, average='macro')

    print('Resultados dataset treino:')
    print(f'Matriz de confusão de treino:\n {cm}')
    print(f'Acurácia na base de treino: {round(acc, 2) * 100}%')
    print(f'Precisão na base de treino: {round(prec, 2) * 100}%')
    print(f'Recall na base de treino: {round(rec, 2) * 100}%')
    print(f'F1 score na base de treino: {round(f1_score, 2) * 100}%')

    # Plot da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Matriz de Confusão: ', weight='bold')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    plt.show()

    plt.figure(figsize=(12, 5))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treinamento')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treinamento e Validação')

    # Plotar a perda de treinamento e validação
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treinamento')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treinamento e Validação')

    plt.show()

def model_predict(model, image_input, type_data=None):

    if type_data == 'image':

        image_resized = cv2.resize(image_input, (28, 28))
        image_normalized = image_resized / 255.0
        image = image_normalized.reshape(1, 28, 28, 1)

    else:
        image = np.expand_dims(image_input, axis=0)

    predictions = model.predict(image)

    predict_result = np.argmax(predictions[0])

    print(f"Result predict: {predict_result}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_input, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.bar(list(range(0, 10)), predictions[0])
    plt.xticks(np.arange(0, 10, 1))
    plt.show()

    return predict_result

# def model_predict_dataset(model, image_input, type_data):
#
#     img_batch = np.expand_dims(image_input, axis=0)
#     predictions = model.predict(img_batch)
#
#     predict_result = np.argmax(predictions[0])
#
#     print(f"Result predict: {predict_result}")
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_input, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.bar(list(range(0, 10)), predictions[0])
#     plt.xticks(np.arange(0, 10, 1))
#     plt.show()
#
#     return predict_result


def create_model():

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
    #                            activation='relu',
    #                            input_shape=(28, 28, 1)),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
    #                            activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    print(model.summary())

    return model

if __name__ == '__main__':

    # Train
    dataset = prepare_dataset_mnist()
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('model_cnn_mnist.keras', monitor='val_accuracy', save_best_only=True)
    history = model.fit(dataset['x_train'], dataset['y_train'], epochs=5, validation_data=(dataset['x_val'], dataset['y_val']), callbacks=[checkpoint])
    model_evaluate(model, dataset, history)
    predict = model_predict(model, dataset['x_test'][random.randint(0, len(dataset['x_test']))])

    # Prediction
    model = load_model('model_cnn_mnist.keras')
    image = cv2.imread('example_mnist.png', 0)
    predict = model_predict(model, image, 'image')
