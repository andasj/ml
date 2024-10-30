import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import viewer as vi
from sklearn import metrics
import warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tensorflow.keras.utils import load_img, img_to_array
import os
import random

img_height = 224
img_width = 224
epochs = 10
batch_size = 32

def prepare_dataset_cats_and_dogs():

    train_dir = 'datasets/cats_and_dogs_filtered/train'
    validation_dir = 'datasets/cats_and_dogs_filtered/validation'

    train_dataset = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_dataset = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_dataset = image_dataset_from_directory(
        validation_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, val_dataset, test_dataset

def create_model():

    normalization_layer = layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    # base_model.trainable = True
    # fine_tune_at = len(base_model.layers) // 2  # Tornar metade das camadas treináveis
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = normalization_layer(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    return model

def model_evaluate(model, test_dataset, history):

    y_pred_probs = model.predict(test_dataset, batch_size=batch_size)
    y_pred = np.squeeze((y_pred_probs > 0.5).astype(int))
    y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)

    cm = metrics.confusion_matrix(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, average='binary')
    rec = metrics.recall_score(y_true, y_pred, average='binary')
    f1_score = metrics.f1_score(y_true, y_pred, average='binary')

    print('Resultados dataset treino:')
    print(f'Matriz de confusão de treino:\n {cm}')
    print(f'Acurácia na base de treino: {round(acc, 2) * 100}%')
    print(f'Precisão na base de treino: {round(prec, 2) * 100}%')
    print(f'Recall na base de treino: {round(rec, 2) * 100}%')
    print(f'F1 score na base de treino: {round(f1_score, 2) * 100}%\n')

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

def prepare_image_predict(image):

    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST_EXACT)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_rgb, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

if __name__ == '__main__':

    # # Train
    # train_dataset, val_dataset, test_dataset = prepare_dataset_cats_and_dogs()
    # model = create_model()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    #
    # # model.save('models/model_cat_and_dog.keras')
    #
    # model_evaluate(model, test_dataset, history)
    # loss_train, acc_train = model.evaluate(train_dataset)
    # loss_val, acc_val = model.evaluate(val_dataset)
    # loss_test, acc_test = model.evaluate(test_dataset)

    # Prediction
    path_dogs = 'datasets/cats_and_dogs_filtered/validation/dogs/'
    path_cats = 'datasets/cats_and_dogs_filtered/validation/cats/'
    path = random.choice([path_dogs, path_cats])
    list_dir = os.listdir(path)

    model = models.load_model('models/model_cat_and_dog.keras')
    result_predict = {0: 'cat', 1: 'dog'}

    img = cv2.imread(path + list_dir[random.randint(0, len(list_dir))])
    img_predict = prepare_image_predict(img)
    predict = (model.predict(img_predict) > 0.5).astype(int)[0, 0]
    vi.show_image([img], [result_predict[predict]])
