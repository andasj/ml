import warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169, Xception, VGG19, NASNetLarge, MobileNetV2, ResNet50V2, EfficientNetB0, InceptionV3, ConvNeXtBase
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import viewer as vi
import cv2

def preprocessing_image(image, model_name):

    if model_name == 'DenseNet169':
        size = (224, 224)

    elif model_name == 'VGG19':
        size = (224, 224)

    elif model_name == 'Xception':
        size = (299, 299)

    elif model_name == 'NASNetLarge':
        size = (331, 331)

    elif model_name == 'MobileNetV2':
        size = (224, 224)

    elif model_name == 'ResNet50V2':
        size = (224, 224)

    elif model_name == 'EfficientNetB0':
        size = (224, 224)

    elif model_name == 'InceptionV3':
        size = (299, 299)

    elif model_name == 'ConvNeXtBase':
        size = (224, 224)

    else:
        print(f'Model {model_name} not find. Use size MobileNetV2.')
        size = (224, 224)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_NEAREST_EXACT)
    img_array = np.array(img_rgb) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, size

def predict_image(model_name, image, size):

    if model_name == 'DenseNet169':
        # model = DenseNet169(weights='imagenet')

        model = DenseNet169(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(model_path)

    elif model_name == 'VGG19':
        # model = VGG19(weights='imagenet')

        model = VGG19(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(model_path)

    elif model_name == 'Xception':
        # model = Xception(weights='imagenet')

        model = Xception(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/xception_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(model_path)

    elif model_name == 'NASNetLarge':
        # model = NASNetLarge(weights='imagenet')

        model = NASNetLarge(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/nasnet_large.h5'
        model.load_weights(model_path)

    elif model_name == 'MobileNetV2':
        # model = MobileNetV2(weights='imagenet')

        model = MobileNetV2(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
        model.load_weights(model_path)

    elif model_name == 'ResNet50V2':
        # model = ResNet50V2(weights='imagenet')

        model = ResNet50V2(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(model_path)

    elif model_name == 'EfficientNetB0':
        # model = EfficientNetB0(weights='imagenet')

        model = EfficientNetB0(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/efficientnetb0.h5'
        model.load_weights(model_path)

    elif model_name == 'InceptionV3':
        # model = InceptionV3(weights='imagenet')

        model = InceptionV3(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(model_path)

    elif model_name == 'ConvNeXtBase':
        # model = ConvNeXtBase(weights='imagenet')

        model = ConvNeXtBase(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/convnext_base.h5'
        model.load_weights(model_path)

    else:
        print(f'Model {model_name} not find. Use MobileNetV2.')
        # model = MobileNetV2(weights='imagenet')

        model = MobileNetV2(weights=None, input_shape=(size[0], size[1], 3))
        model_path = 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
        model.load_weights(model_path)

    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    predicted_label = decoded_predictions[1]

    return predicted_label

if __name__ == '__main__':

    image_path = 'images/flamingo.jpg'
    image = cv2.imread(image_path)
    model_name_dict = {0:'DenseNet169', 1:'VGG19', 2:'Xception', 3:'NASNetLarge', 4:'MobileNetV2', 5:'ResNet50V2', 6:'EfficientNetB0', 7:'InceptionV3', 8:'ConvNeXtBase'}
    model_name = model_name_dict[8]

    image_predict, size = preprocessing_image(image=image, model_name=model_name)
    predicted_label = predict_image(model_name=model_name, image=image_predict, size=size)

    vi.show_image([image], [predicted_label])
