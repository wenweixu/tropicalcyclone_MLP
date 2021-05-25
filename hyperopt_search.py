from hyperopt import fmin, tpe, STATUS_OK, Trials
import numpy as np
#np.random.seed(10)
import utils
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Lambda, Dense, Input, Conv2D, BatchNormalization, Flatten, Concatenate, Dropout, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import sys
import os
import datetime
import space_declarations
from sklearn.model_selection import train_test_split
from keras import regularizers


# fit globals
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.05, patience=10, restore_best_weights=True)]
fit_kwargs = {'epochs': 2000,
              'verbose': 2,
              'callbacks': callbacks}

def mlp_create_model(space):
    # make model
    input_ = Input(shape=x_train[0].shape)
    x = Dense(space['n_nodes_layer1'], activation=space['layer1_activation'])(input_)
    if space['num_layers']['layers'] == 'two':
        x = Dense(space['num_layers']['nodes2'], activation=space['num_layers']['activation2'])(x)
    elif space['num_layers']['layers'] == 'three':
        x = Dense(space['num_layers']['nodes2'], activation=space['num_layers']['activation2'])(x)
        x = Dense(space['num_layers']['nodes3'], activation=space['num_layers']['activation3'])(x)
    elif space['num_layers']['layers'] == 'four':
        x = Dense(space['num_layers']['nodes2'], activation=space['num_layers']['activation2'])(x)
        x = Dense(space['num_layers']['nodes3'], activation=space['num_layers']['activation3'])(x)
        x = Dense(space['num_layers']['nodes4'], activation=space['num_layers']['activation4'])(x)
    elif space['num_layers']['layers'] == 'five':
        x = Dense(space['num_layers']['nodes2'], activation=space['num_layers']['activation2'])(x)
        x = Dense(space['num_layers']['nodes3'], activation=space['num_layers']['activation3'])(x)
        x = Dense(space['num_layers']['nodes4'], activation=space['num_layers']['activation4'])(x)
        x = Dense(space['num_layers']['nodes5'], activation=space['num_layers']['activation5'])(x)
    x = Dense(1, activation='linear')(x)
    model = Model(inputs=input_, outputs=x)
    model.compile(optimizer=Adam(lr=space['learning_rate']), loss='mean_squared_error', metrics=['mae'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=space['batch_size'], **fit_kwargs)

    val_mean_absolute_error = model.evaluate(x_val, y_val)[1]

    space['val_mean_absolute_error'] = val_mean_absolute_error
    print(f'\n{space}')
    print('Best val mean absolute error of epoch:', val_mean_absolute_error)

    # colnames for csv
    col_names = ['val_mean_absolute_error', 'learning_rate', 'batch_size', 'layers', 'n_nodes_layer1',
                'layer1_activation', 'nodes2', 'activation2', 'nodes3', 'activation3', 'nodes4', 'activation4',
                'nodes5', 'activation5']

    # write the csv header
    if 'filename' not in globals():
        # create directory
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, 'search_results')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        global filename
        filename = 'search_results/mlp' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.csv'
        with open(filename, 'a+') as f:
            line = ','.join(col_names)
            f.write(line + '\n')

    # write results to csv
    with open(filename, 'a+') as f:
        for name in col_names:
            if name == 'val_mean_absolute_error':
                line = val_mean_absolute_error
            elif name in space.keys():
                line = space[name]
            elif name in space['num_layers'].keys():
                line = space['num_layers'][name]
            else:
                line = ''
            f.write(str(line) + ',')
        f.write('\n')

    return {'loss': val_mean_absolute_error, 'status': STATUS_OK}


def cnn_create_model(space):
    common_args = {'activation':'relu', 'padding':'same'}
    input = Input(x_train[0].shape)
    if space['num_layers']['layers'] == 'one':
        x = Conv2D(space['num_layers']['n_convs1'], space['num_layers']['kernal_sz1'], **common_args)(input)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

    if space['num_layers']['layers'] == 'two':
        x = Conv2D(space['num_layers']['n_convs1'], space['num_layers']['kernal_sz1'], **common_args)(input)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs2'], space['num_layers']['kernal_sz2'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

    if space['num_layers']['layers'] == 'three':
        x = Conv2D(space['num_layers']['n_convs1'], space['num_layers']['kernal_sz1'], **common_args)(input)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs2'], space['num_layers']['kernal_sz2'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs3'], space['num_layers']['kernal_sz3'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

    if space['num_layers']['layers'] == 'four':
        x = Conv2D(space['num_layers']['n_convs1'], space['num_layers']['kernal_sz1'], **common_args)(input)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs2'], space['num_layers']['kernal_sz2'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs3'], space['num_layers']['kernal_sz3'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

        x = Conv2D(space['num_layers']['n_convs4'], space['num_layers']['kernal_sz4'], **common_args)(x)
        x = MaxPool2D()(x)
        if space['batch_norm']:
            x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    # x = Lambda(lambda z: z * 10 * np.std(y_train))(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=space['learning_rate']), loss='mean_squared_error', metrics=['mae'])

    dataflow = ImageDataGenerator(horizontal_flip=True, vertical_flip=True).flow(x_train, y_train)

    model.fit_generator(dataflow,
                        validation_data=(x_val, y_val),
                        steps_per_epoch=(len(y_train) // 32) + 1,
                        **fit_kwargs)

    val_mean_absolute_error = model.evaluate(x_val, y_val)[1]

    space['val_mean_absolute_error'] = val_mean_absolute_error
    print(f'\n{space}')
    print('Best val mean absolute error of epoch:', val_mean_absolute_error)

    # colnames for csv
    col_names = ['val_mean_absolute_error', 'learning_rate', 'batch_norm',
                 'n_convs1', 'n_convs2', 'n_convs3', 'n_convs4',
                 'kernal_sz1', 'kernal_sz2', 'kernal_sz3', 'kernal_sz4']

    # write the csv header
    if 'filename' not in globals():
        # create directory
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r'search_results')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        global filename
        filename = 'search_results/cnn' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.csv'
        with open(filename, 'a+') as f:
            line = ','.join(col_names)
            f.write(line + '\n')

    # write results to csv
    with open(filename, 'a+') as f:
        for name in col_names:
            if name == 'val_mean_absolute_error':
                line = val_mean_absolute_error
            elif name in space.keys():
                line = space[name]
            elif name in space['num_layers'].keys():
                if type(space['num_layers'][name]) == tuple:
                    line = space['num_layers'][name][0]
                else:
                    line = space['num_layers'][name]
            else:
                line = ''
            f.write(str(line) + ',')
        f.write('\n')

    return {'loss': val_mean_absolute_error, 'status': STATUS_OK}


def cnn_augmented_create_model(space):
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.05, patience=15, restore_best_weights=True)]

    fit_kwargs = {'epochs': 3000,
                  'verbose': 0,
                  'callbacks': callbacks}

    image_input = Input(shape=x_train[0][0].shape)
    x = Conv2D(space['n_nodes_layer1'], (4,4), strides=2, kernel_initializer='glorot_uniform', padding='same', activation=space['layer1_activation'])(image_input)
    x = BatchNormalization()(x)
    x = Conv2D(space['n_nodes_layer2'], (3,3), strides=2, kernel_initializer='glorot_uniform', padding='same', activation=space['layer2_activation'])(x)
    x = BatchNormalization()(x)
    x = Conv2D(space['n_nodes_layer3'], (3,3), strides=2, kernel_initializer='glorot_uniform', padding='same', activation=space['layer3_activation'])(x)
    x = BatchNormalization()(x)
    x = Conv2D(space['n_nodes_layer4'], (3,3), strides=2, kernel_initializer='glorot_uniform', padding='same', activation=space['layer4_activation'])(x)
    x = BatchNormalization()(x)
    x = Conv2D(space['n_nodes_layer5'], (1,1), kernel_initializer='glorot_uniform', padding='same', activation=space['layer5_activation'])(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    hand_input = Input(shape=x_train[1][0].shape)
    h = Dense(space['n_nodes_layer6'], activation=space['layer6_activation'], kernel_initializer='glorot_uniform')(hand_input)
    h = Dense(space['n_nodes_layer7'], activation=space['layer7_activation'], kernel_initializer='glorot_uniform')(h)
    h = Dense(space['n_nodes_layer8'], activation=space['layer8_activation'], kernel_initializer='glorot_uniform')(h)

    x = Concatenate()([x, h])
    x = Dense(space['n_nodes_layer9'], activation=space['layer9_activation'], kernel_initializer='glorot_uniform')(x)
    x = Dense(space['n_nodes_layer10'], activation=space['layer10_activation'], kernel_initializer='glorot_uniform')(x)
    x = Dense(1, activation='linear')(x)
    x = Lambda(lambda z: z*10*np.std(y_train))(x)
    model = Model(inputs=[image_input,hand_input], outputs=x)
    model.compile(optimizer=Adam(lr=space['learning_rate']), loss='mean_squared_error', metrics=['mae'])

    result = model.fit(x_train, y_train, batch_size=space['batch_size'],
              validation_data=(x_val, y_val),
              **fit_kwargs)

    # get the lowest val_mean_absolute_error of the training epochs
    val_mean_absolute_error = np.amin(result.history['val_mean_absolute_error'])
    space['val_mean_absolute_error'] = val_mean_absolute_error
    print(f'\n{space}')
    print('Best val mean absolute error of epoch:', val_mean_absolute_error)

    # colnames for csv
    colNames = ['val_mean_absolute_error', 'learning_rate', 'batch_size', 'n_nodes_layer1', 'layer1_activation',
                'n_nodes_layer2', 'layer2_activation', 'n_nodes_layer3', 'layer3_activation',
                'n_nodes_layer4', 'layer4_activation', 'n_nodes_layer5', 'layer5_activation',
                'n_nodes_layer6', 'layer6_activation', 'n_nodes_layer7', 'layer7_activation',
                'n_nodes_layer8', 'layer8_activation', 'n_nodes_layer9', 'layer9_activation',
                'n_nodes_layer10', 'layer10_activation']

    # write the csv header
    if 'filename' not in globals():
        # create directory
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r'search_results')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        global filename
        filename = 'search_results/cnn_augmented' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.csv'
        with open(filename, 'a+') as f:
            line = ','.join(colNames)
            f.write(line + '\n')

    # write results to csv
    with open(filename, 'a+') as f:
        for name in colNames:
            if name == 'val_mean_absolute_error':
                line = val_mean_absolute_error
            elif name in space.keys():
                line = space[name]
            elif name in space['num_layers'].keys():
                line = space['num_layers'][name]
            else:
                line = ''
            f.write(str(line) + ',')
        f.write('\n')

    return {'loss': val_mean_absolute_error, 'status': STATUS_OK, 'model': model}

def fused_model(space):
    np.random.seed(space['np_seed'])
    tf.set_random_seed(space['tf_seed'])
    # make model
    input_ = Input(shape=x_train[0].shape)
    x = Dense(space['n_nodes_layer1'], activation=space['layer1_activation'])(input_)
    if space['num_layers']['layers'] == 'two':
        x = Dense(space['num_layers']['nodes2'], activation=space['num_layers']['activation2'], kernel_regularizer=regularizers.l2(space['l2_1']))(x)
    x = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(space['l2_2']))(x)
    model = Model(inputs=input_, outputs=x)
    model.compile(optimizer=Adam(lr=space['learning_rate']), loss=space['loss'], metrics=['mae'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=space['batch_size'], **fit_kwargs)

    val_mean_absolute_error = model.evaluate(x_val, y_val)[1]
    test_mean_absolute_error = model.evaluate(x_test, y_test)[1]

    space['val_mean_absolute_error'] = val_mean_absolute_error
    space['test_mean_absolute_error'] = test_mean_absolute_error
    print(f'\n{space}')
    print('Best val mean absolute error of epoch:', val_mean_absolute_error)

    # colnames for csv
    col_names = ['val_mean_absolute_error', 'test_mean_absolute_error', 'learning_rate', 'batch_size', 'layers', 'n_nodes_layer1',
                'layer1_activation', 'nodes2', 'activation2', 'nodes3', 'activation3', 'nodes4', 'activation4',
                'nodes5', 'activation5','loss', 'l2_1', 'l2_2', 'np_seed','tf_seed']

    # write the csv header
    if 'filename' not in globals():
        # create directory
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, 'search_results')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        global filename
        filename = 'search_results/fused' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.csv'
        with open(filename, 'a+') as f:
            line = ','.join(col_names)
            f.write(line + '\n')

    # write results to csv
    with open(filename, 'a+') as f:
        for name in col_names:
            if name in space.keys():
                line = space[name]
            elif name in space['num_layers'].keys():
                line = space['num_layers'][name]
            else:
                line = ''
            f.write(str(line) + ',')
        f.write('\n')

    return {'loss': test_mean_absolute_error, 'status': STATUS_OK} #val_mean_absolute_error

if __name__ == '__main__':
    '''
    example: python hyperopt_search.py fused 2>&1 | tee search.log
    '''
    # input
    architecture = sys.argv[1]
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # config.gpu_options.allow_growth = True
    from keras.backend.tensorflow_backend import set_session
    tf.set_random_seed(11)
    set_session(tf.Session(config=config))

    # number of different evaluation attempts tried
    max_evals = 500

    if architecture == 'mlp':
        x_train, x_val, y_train, y_val, _ = utils.load_hand_data_cv()
        space = space_declarations.mlp_space
        trials = Trials()
        best = fmin(mlp_create_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    elif architecture == 'cnn':
        x_train, x_val, y_train, y_val, _ = utils.load_image_data_cv()
        space = space_declarations.cnn_space
        trials = Trials()
        best = fmin(cnn_create_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    elif architecture == 'cnn_augmented':
        x_train, x_test, y_train, y_test, all_train_ids = utils.load_data(get_images=True, get_hand=True, scale=True)
        space = space_declarations.cnn_augmented_space
        trials = Trials()
        best = fmin(cnn_augmented_create_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    elif architecture == 'fused':
        x_train_all, x_test, y_train_all, y_test, all_train_ids = utils.load_augmented_features()
        unique_train_ids = np.unique(all_train_ids)
        train_cv_ids, val_ids = train_test_split(unique_train_ids,test_size=0.08)
        train_idxs = np.isin(all_train_ids, train_cv_ids)
        val_idxs   = np.isin(all_train_ids, val_ids)
        print(('training size: ', train_cv_ids.shape))
        print(('validating size: ', val_ids.shape))
        x_train = x_train_all[train_idxs]
        y_train = y_train_all[train_idxs]
        x_val = x_train_all[val_idxs]
        y_val = y_train_all[val_idxs]
        space = space_declarations.fused_space
        trials = Trials()
        best = fmin(fused_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    else:
        raise Exception(f'Invalid architecture name: {architecture}')

    print(best)
