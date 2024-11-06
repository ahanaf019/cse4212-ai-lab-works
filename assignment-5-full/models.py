import tensorflow as tf
import keras


def build_simple_cnn(image_size) -> keras.Model:
    inputs = keras.Input(shape=image_size + (3,))
    x = keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')(inputs)
    x = keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name='SCNN_Model')


def build_fcnn(image_size) -> keras.Model:
    inputs = keras.Input(shape=image_size + (3,))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name='FC_Model')


def build_vgg16(image_size, weights=None) -> keras.Model:
    vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights=weights, input_shape=image_size+(3,))

    inputs = vgg16_model.input
    x = keras.layers.Flatten()(vgg16_model.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name='VGG16_Model')
