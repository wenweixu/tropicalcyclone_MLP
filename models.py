import numpy as np
#np.random.seed(12)
import tensorflow as tf
#tf.set_random_seed(11)
from keras.layers import Dense, Input, Conv2D, BatchNormalization, Flatten, Concatenate, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras import regularizers

# def mlp(input_shape=None):
#     input = Input(shape=input_shape)
#     x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(input)
#     x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(16, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(1, activation='linear', kernel_initializer='glorot_uniform')(x)
#     model = Model(inputs=input, outputs=x)
#     model.compile(optimizer=Adam(lr=0.0005), loss='mse', metrics=['mae'])
#     return model
#


def mlp(input_shape=None):
    input = Input(shape=input_shape)
    x = Dense(2048, activation='sigmoid')(input)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    #x = Lambda(lambda z: z * y_train_std)(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001), loss='mae', metrics=['mae'])
    return model


def mlp_augmented_features(input_shape=None):
    '''
    mlp architecture trained on features from globally trained cnn and globally trained mlp
    '''
    input = Input(shape=input_shape)
    x = Dense(16, activation='sigmoid')(input)
    # # #x = Dense(16, activation='relu')(x)
    # x = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=0.01), loss='mae', metrics=['mae'])
    return model


def cnn(input_shape=None):
    '''
    cnn architecture found via hyperopt
    '''
    common_args = {'padding':'same',
                   'activation':'relu'}

    input = Input(shape=input_shape)
    x = Conv2D(32, (8,8), **common_args)(input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (4,4), **common_args)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (2,2), **common_args)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (8,8), **common_args)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])
    return model


# def cnn_augmented(image_shape=None, hand_shape=None):
#     '''
#     Augmented cnn for training on aggregated data (non-global). Uses the cnn architecture from hyperopt.
#     '''
#     common_args = {'padding': 'same',
#                    'activation': 'relu'}
#
#     image_input = Input(shape=image_shape)
#     x = Conv2D(32, (8, 8), **common_args)(image_input)
#     x = MaxPool2D()(x)
#     x = BatchNormalization()(x)
#
#     x = Conv2D(32, (4, 4), **common_args)(x)
#     x = MaxPool2D()(x)
#     x = BatchNormalization()(x)
#
#     x = Conv2D(128, (2, 2), **common_args)(x)
#     x = MaxPool2D()(x)
#     x = BatchNormalization()(x)
#
#     x = Conv2D(32, (8, 8), **common_args)(x)
#     x = MaxPool2D()(x)
#     x = BatchNormalization()(x)
#     x = Flatten()(x)
#
#     hand_input = Input(shape=hand_shape)
#     h = Dense(300, activation='relu', kernel_initializer='glorot_uniform')(hand_input)
#     h = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(h)
#     h = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(h)
#
#     x = Concatenate()([x, h])
#     x = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(x)
#     x = Dense(1, activation='linear')(x)
#     # x = Lambda(lambda z: z*10*y_train_std)(x)
#     model = Model(inputs=[image_input,hand_input], outputs=x)
#     model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae'])
#     return model
