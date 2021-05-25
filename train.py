import sys
import numpy as np
#np.random.seed(4)
import tensorflow as tf
#tf.set_random_seed(12)
import utils
import models
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
'''
Do cross validation training on the various NN models.

Usage:
python train.py mlp model_mlp.model
# --architecture mlp_augmented_features \
# --gpu_id 5

'''

# parser = ArgumentParser()
# parser.add_argument('--architecture')
# parser.add_argument('--gpu_id')
# parser.add_argument('--model_save_filename', help='If this is set then the model will be saved after the first fold of training, then the script will exit.')
# args = parser.parse_args()
architecture = sys.argv[1]
model_save_filename = sys.argv[2]
#architecture = args.architecture
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# from keras.backend.tensorflow_backend import set_session
# set_session(tf.Session(config=config))


callbacks = [EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.05, patience=10, restore_best_weights=True)]
#callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.05, patience=15, restore_best_weights=True)]

fit_kwargs = {'epochs': 1000,
              'verbose': 2,
              'batch_size':64,
              'callbacks': callbacks}

# Load data
print('Loading data...')
if architecture == 'mlp':
    x_train, x_test, y_train, y_test, all_train_ids = utils.load_hand_data_cv()
elif architecture == 'cnn':
    x_train, x_test, y_train, y_test, all_train_ids = utils.load_image_data_cv()
elif architecture == 'cnn_augmented':
    x_train, x_test, y_train, y_test, all_train_ids = utils.load_augmented_data_cv()
elif architecture == 'mlp_augmented_features':
    x_train, x_test, y_train, y_test, all_train_ids = utils.load_augmented_features()
else:
    raise Exception(f'Invalid architecture name: {architecture}')
unique_train_ids = np.unique(all_train_ids)

# Init CV
n_splits = 5
metrics = {'MAE': [],
           'RMSE': [],
           'R^2': []}

# Cross validation loop
for i in range(n_splits):
    print(f'\n--- Fold {i+1} of {n_splits} ---')

    # Split fold on hurricanes: don't let the same hurricane be in tain and val
    train_cv_ids, val_ids = train_test_split(unique_train_ids,test_size=0.08)
    train_idxs = np.isin(all_train_ids, train_cv_ids)
    val_idxs   = np.isin(all_train_ids, val_ids)
    print(('training size: ', train_cv_ids.shape))
    print(('validating size: ', val_ids.shape))

    if architecture == 'mlp':
        # Extract CV fold
        x_train_cv = x_train[train_idxs]
        y_train_cv = y_train[train_idxs]
        x_val = x_train[val_idxs]
        y_val = y_train[val_idxs]
        # Train
        model = models.mlp(input_shape=x_train_cv[0].shape)
        print(model.summary())
        model.fit(x_train_cv, y_train_cv,
                  validation_data=(x_val, y_val),
                  **fit_kwargs)
        utils.save_model(model,model_save_filename)
        y_predict = model.predict(x_test)

    elif architecture == 'mlp_augmented_features':
        # np.random.seed(0)
        # tf.set_random_seed(12)
        # Extract CV fold
        x_train_cv = x_train[train_idxs]
        y_train_cv = y_train[train_idxs]
        x_val = x_train[val_idxs]
        y_val = y_train[val_idxs]
        # Train
        model = models.mlp_augmented_features(input_shape=x_train_cv[0].shape)
        print(model.summary())
        model.fit(x_train_cv, y_train_cv,
                  validation_data=(x_val, y_val),
                  **fit_kwargs)
        utils.save_model(model,model_save_filename)
        y_predict = model.predict(x_test)

    elif architecture == 'cnn':
        # Extract CV fold
        x_train_cv = x_train[train_idxs]
        y_train_cv = y_train[train_idxs]
        x_val = x_train[val_idxs]
        y_val = y_train[val_idxs]
        # Train
        model = models.cnn(input_shape=x_train_cv[0].shape)
        dataflow = ImageDataGenerator(horizontal_flip=True, vertical_flip=True).flow(x_train_cv, y_train_cv)
        model.fit_generator(dataflow,
                            validation_data=(x_val, y_val),
                            steps_per_epoch=(len(y_train_cv) // 32) + 1,
                            **fit_kwargs)
        utils.save_model(model,model_save_filename)
        y_predict = utils.predict_with_rotations(model, x_test, architecture=architecture)

    elif architecture == 'cnn_augmented':
        # Extract CV fold
        x_train_cv = [x_train[0][train_idxs], x_train[1][train_idxs]]
        y_train_cv = y_train[train_idxs]
        x_val = [x_train[0][val_idxs], x_train[1][val_idxs]]
        y_val = y_train[val_idxs]
        # Train
        model = models.cnn_augmented(image_shape=x_train_cv[0][0].shape, hand_shape=x_train_cv[1][0].shape)
        model.fit_generator(utils.image_generator(x_train_cv, y_train_cv),
                            validation_data=(x_val, y_val),
                            steps_per_epoch=(len(y_train_cv) // 32) + 1,
                            shuffle=False,
                            **fit_kwargs)
        utils.save_model(model,model_save_filename)
        y_predict = utils.predict_with_rotations(model, x_test, architecture=architecture)

    tmp_metrics = utils.compute_metrics(y_test, y_predict, print_them=True)
    metrics = {k:v+[tmp_metrics[k]] for k,v in metrics.items()}


# Print metrics
print('\n--- Cross Validation Test Metrics ---')
for k, v in metrics.items():
    print(f'{k}: {np.mean(v):.2f} +/- {2 * np.std(v):.2f}')
