import warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns
import viewer as vi
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']
img_height = 32
img_width = 32

def prepare_dataset_cifar10():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      random_state=42)

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
        img_rescaled = (x_train[i] + 1) / 2.0
        plt.imshow(img_rescaled, cmap=plt.cm.binary)
        plt.xlabel(class_names[list(y_train[i]).index(max(list(y_train[i])))])
    plt.show()

    return dict_dataset

def model_evaluate(model, dataset, history):

    y_pred_probs = model.predict(dataset['x_test'])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_dataset = np.argmax(dataset['y_test'], axis=1)

    cm = metrics.confusion_matrix(y_dataset, y_pred)
    acc = metrics.accuracy_score(y_dataset, y_pred)
    prec = metrics.precision_score(y_dataset, y_pred, average='macro')
    rec = metrics.recall_score(y_dataset, y_pred, average='macro')
    f1_score = metrics.f1_score(y_dataset, y_pred, average='macro')

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

def create_model():
    # Carregar o modelo MobileNet pré-treinado
    mobilenet = MobileNet(weights='imagenet', include_top=False,
                          input_shape=(img_height, img_width, 3))

    # mobilenet.trainable = True

    # Descongelar as últimas camadas do MobileNet para fine-tuning
    for layer in mobilenet.layers[-10:]:
        layer.trainable = True

    # Construir o modelo
    model = models.Sequential([
        mobilenet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    print(model.summary())

    return model

def prepare_image_predict(image):

    img_resized = cv2.resize(image, (img_height, img_width), interpolation=cv2.INTER_NEAREST_EXACT)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype('float32')
    img_rgb = preprocess_input(img_rgb)
    img_array = np.expand_dims(img_rgb, axis=0)

    return img_array, img_resized

if __name__ == '__main__':

    # Train
    dataset = prepare_dataset_cifar10()
    model = create_model()

    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(dataset['x_train'])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('models/model_cifar10.keras', monitor='val_accuracy', save_best_only=True)
    history = model.fit(datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=32), validation_data=(dataset['x_val'], dataset['y_val']), epochs=100, callbacks=[checkpoint])

    model_evaluate(model, dataset, history)
    train_loss, train_accuracy = model.evaluate(dataset['x_train'], dataset['y_train'])
    val_loss, val_accuracy = model.evaluate(dataset['x_val'], dataset['y_val'])
    test_loss, test_accuracy = model.evaluate(dataset['x_test'], dataset['y_test'])

    # # Prediction
    # model = models.load_model('models/model_cifar10.keras')
    #
    # path_test = 'datasets/resized_cifar10/test/'
    # path_class = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])
    # list_dir = os.listdir(path_test + str(path_class) + '/')
    #
    # image = cv2.imread(path_test + str(path_class) + '/' + list_dir[random.randint(0, len(list_dir))])
    #
    # image_predict, image_show = prepare_image_predict(image)
    # predict = model.predict(image_predict)
    # print(class_names)
    # print(predict[0])
    # vi.show_image([image, image_show], ['input', class_names[np.argmax(predict[0])]])


