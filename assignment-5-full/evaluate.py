import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '' 
# https://github.com/tensorflow/tensorflow/issues/53519
os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '256' 

import keras
import tensorflow as tf
from models import *
import matplotlib.pyplot as plt
import subprocess

results = {}

def main():
    dataset_base_path = '.'
    db_name = 'dataset_256'
    
    epochs = 50
    image_size = (256, 256)
    weights = 'imagenet'
    lr_decay_factor = 0.1
    
    # batch_size = 16
    # init_lr = 1e-4
    # model_name = f'simple_cnn'

    subprocess.run(f'unzip -o -q {dataset_base_path}/{db_name}.zip', shell=True, capture_output=True, text=True)

    batch_size = 32
    train_ds, val_ds, test_ds, class_names = load_dataset(
        path=f'{dataset_base_path}/{db_name}', 
        subdirs=['train', 'valid', 'test'], 
        image_size=image_size, 
        batch_size=batch_size, 
        label_mode='int'
    )
    
    model_name = 'vgg16256'
    bs = 16
    lr = 1e-3
    
    model = keras.models.load_model(f'./checkpoints/{model_name}_{bs}__{lr}__imagenet.keras')
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.F1Score(name='f1score'),
        ]
    )
    
    preprocess = keras.applications.vgg16.preprocess_input
    test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(lambda x, y: (x, tf.cast(y, tf.float32)))
    # test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    eval = model.evaluate(test_ds)
    print(eval)


def experiment(model_name, dataset_base_path, db_name, image_size, batch_size, init_lr,
                weights, lr_decay_factor, epochs):
    train_ds, val_ds, test_ds, class_names = load_dataset(
        path=f'{dataset_base_path}/{db_name}', 
        subdirs=['train', 'valid', 'test'], 
        image_size=image_size, 
        batch_size=batch_size, 
        label_mode='int'
    )
    
    mn = model_name
    
    if model_name == 'simple_cnn':
        model = build_simple_cnn(image_size=image_size)
        model_name = f'{model_name}{image_size[0]}_{batch_size}__{init_lr}'
    
    if model_name == 'fcnn':
        model = build_fcnn(image_size=image_size)
        model_name = f'{model_name}{image_size[0]}_{batch_size}__{init_lr}'
    
    if model_name == 'vgg16':
        model = build_vgg16(image_size=image_size, weights=weights)
        model_name = f'{model_name}{image_size[0]}_{batch_size}__{init_lr}__{weights}'
        
    if mn == 'vgg16' and weights == 'imagenet':
        for layer in model.layers[:-4]:
            layer.trainable = False
        
        print('Freezing weights and applying preprocessing')
        preprocess = keras.applications.vgg16.preprocess_input
        train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(f'='*30)
    print(f'Training: {model_name}')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=init_lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
        ]
    )
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(factor=0.01, patience=2, verbose=1),
        CustomLearningRateSchedule(decay_factor=lr_decay_factor, initial_lr=init_lr),
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f'checkpoints/{model_name}.keras', save_best_only=True, save_weights_only=False),
        keras.callbacks.CSVLogger(f'logs/{model_name}.csv', append=True),
    ]
    # model.summary()
    
    history = model.fit(
        train_ds,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=val_ds,
    )
    
    model.load_weights(f'checkpoints/{model_name}.keras')
    eval = model.evaluate(val_ds, return_dict=True)
    results[model_name] = eval
    
    with open('results_dynamic.txt', 'a+') as file:
        file.write(f'{model_name}\n{eval}\n\n')
    
    if weights == 'imagenet':
        model_name = f'{model_name}_fullTL'
        for layer in model.layers:
            layer.trainable = True
        print(f'='*30)
        print(f'Training: {model_name}')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=init_lr),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
            ]
        )
        
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(factor=0.01, patience=2, verbose=1),
            CustomLearningRateSchedule(decay_factor=lr_decay_factor, initial_lr=init_lr),
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(f'checkpoints/{model_name}.keras', save_best_only=True, save_weights_only=False),
            keras.callbacks.CSVLogger(f'logs/{model_name}.csv', append=True),
        ]
        # model.summary()
        
        history = model.fit(
            train_ds,
            callbacks=callbacks,
            epochs=epochs,
            validation_data=val_ds,
        )
        
        model.load_weights(f'checkpoints/{model_name}.keras')
        eval = model.evaluate(val_ds, return_dict=True)
        results[model_name] = eval
        
        with open('results_dynamic.txt', 'a+') as file:
            file.write(f'{model_name}\n{eval}\n\n')
    
    del train_ds
    del val_ds
    del test_ds
    del model


def load_dataset(path='', subdirs=['train', 'val', 'test'], image_size=None, batch_size=None, label_mode=None):
    train_ds = keras.utils.image_dataset_from_directory(
        f'{path}/{subdirs[0]}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=True
    )

    val_ds = keras.utils.image_dataset_from_directory(
        f'{path}/{subdirs[1]}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False
    )

    test_ds = keras.utils.image_dataset_from_directory(
        f'{path}/{subdirs[2]}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False
    )

    # Used for plotting
    class_names = train_ds.class_names
    test_ds = test_ds.map(lambda x, y: (x, tf.cast(y, tf.float32)))
    return train_ds, val_ds, test_ds, class_names


class CustomLearningRateSchedule(keras.callbacks.Callback):
    def __init__(self, initial_lr, decay_factor=0.1):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor

    def on_epoch_begin(self, epoch, logs=None):
            lr = self.model.optimizer.lr
            lr = self.initial_lr * tf.exp(-self.decay_factor * epoch)
            self.model.optimizer.lr = lr

if __name__ == '__main__':
    main()