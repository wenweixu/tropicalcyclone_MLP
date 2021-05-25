import sys
import numpy as np
np.random.seed(int(sys.argv[2]))
import tensorflow as tf
tf.set_random_seed(int(sys.argv[3]))
import utils
import models
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd

import os
import datetime

'''
run script:
python loyo_testing.py <architecture> <numpy seed>  <tensorflow seed>
for example:
python loyo_testing.py mlp 1 1 
'''


if __name__ == '__main__':
    # input
    architecture = sys.argv[1]

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, restore_best_weights=True)]

    fit_kwargs = {'epochs': 1000,
                  'verbose': 2,
                  'callbacks': callbacks}

    # define plot lists
    x_plot = []
    y_plot = []
    y_error = []

    # write the csv header
    colNames = ['Leave Out Year', 'MAE', 'RMSE', 'R^2']
    # create directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'LOYO_results', r'seeds')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    #filename = 'LOYO_results/LOYO' + str(datetime.datetime.now()).replace(' ', '_').replace(':','.') + '.csv'
    filename = 'LOYO_results/seeds/LOYO_np' +str(sys.argv[2])  +'_tf' +str(sys.argv[3]) + '.csv'
    with open(filename, 'a+') as f:
        line = ','.join(colNames) + ','
        f.write(line + '\n')

    for leave_out_year in range(2010, 2019): #(2020, 2021) <<<< update years here
        # Load data
        print(f'Loading data for year: {leave_out_year}...')
        if architecture == 'mlp':
            x_train, x_test, y_train, y_test, ids= utils.load_loyo_data(leave_out_year, scale=True, get_hand=True, remove_oprreadup=False,remove_oprfortraining=True)
        elif architecture == 'cnn':
            x_train, x_test, y_train, y_test, ids = utils.load_loyo_data(leave_out_year, get_images=True, scale=True)
        elif architecture == 'cnn_augmented':
            x_train, x_test, y_train, y_test, ids = utils.load_loyo_data(leave_out_year, get_images=True, get_hand=True, scale=True)
        else:
            raise Exception(f'Invalid architecture name: {architecture}')
        train_hurricane_names = ids

        # Init CV
        n_splits = 1
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.1)
        metrics = {'MAE': [],
                   'RMSE': [],
                   'R^2': []}

        # Cross validation loop
        for i, (train_idxs, val_idxs) in enumerate(ss.split(train_hurricane_names)):
            print(f'\n--- Fold {i+1} of {n_splits} ---')

            if architecture == 'mlp':
                # Extract CV fold
                x_train_cv = x_train[train_idxs]
                y_train_cv = y_train[train_idxs]
                x_val = x_train[val_idxs]
                y_val = y_train[val_idxs]
                # Train
                model = models.mlp(input_shape=x_train_cv[0].shape)
                model.fit(x_train_cv, y_train_cv, batch_size=32,
                          validation_data=(x_val, y_val),
                          **fit_kwargs)
                #utils.save_model(model,'mlp_new.h5')
                y_predict = model.predict(x_test)

            elif architecture == 'cnn':
                # Extract CV fold
                x_train_cv = x_train[train_idxs]
                y_train_cv = y_train[train_idxs]
                x_val = x_train[val_idxs]
                y_val = y_train[val_idxs]
                # Train
                model = models.cnn(input_shape=x_train_cv[0].shape, y_train_std=np.std(y_train_cv))
                dataflow = ImageDataGenerator(horizontal_flip=True, vertical_flip=True).flow(x_train_cv, y_train_cv)
                model.fit_generator(dataflow,
                                    validation_data=(x_val, y_val),
                                    steps_per_epoch=(len(y_train_cv) // 32) + 1,
                                    **fit_kwargs)
                y_predict = utils.predict_with_rotations(model, x_test, architecture=architecture)

            elif architecture == 'cnn_augmented':
                # Extract CV fold
                x_train_cv = [x_train[0][train_idxs], x_train[1][train_idxs]]
                y_train_cv = y_train[train_idxs]
                x_val = [x_train[0][val_idxs], x_train[1][val_idxs]]
                y_val = y_train[val_idxs]
                # Train
                model = models.cnn_augmented(image_shape=x_train_cv[0][0].shape, hand_shape=x_train_cv[1][0].shape, y_train_std=np.std(y_train_cv))
                model.fit_generator(utils.image_generator(x_train_cv, y_train_cv),
                                    validation_data=(x_val, y_val),
                                    steps_per_epoch=(len(y_train_cv) // 32) + 1,
                                    shuffle=False,
                                    **fit_kwargs)
                y_predict = utils.predict_with_rotations(model, x_test, architecture=architecture)

            tmp_metrics = utils.compute_metrics(y_test, y_predict, print_them=True)
            metrics = {k: v+[tmp_metrics[k]] for k, v in metrics.items()}

            # output predicted labels along with true labels for the RI analysis
            df_pred = pd.DataFrame({'y_test':list(y_test.reshape([-1,])), 'y_predict':list(y_predict.reshape([-1,]))})
            pred_filename = 'LOYO_results/seeds/LOYO_pred_np' +str(sys.argv[2])  +'_tf' +str(sys.argv[3]) + str(leave_out_year) + '.csv'
            df_pred.to_csv(pred_filename, index=False)

        # Print metrics
        print(f'\n--- Cross Validation Test Metrics for year: {leave_out_year} ---')
        for k, v in metrics.items():
            print(f'{k}: {np.mean(v):.2f} +/- {2 * np.std(v):.2f}')

        # write results to csv
        with open(filename, 'a+') as f:
            for name in colNames:
                if name == 'Leave Out Year':
                    line = leave_out_year
                else:
                    line = f'{np.mean(metrics[name]):.2f} +/- {2 * np.std(metrics[name]):.2f}'
                    # update plot values
                    if name == "MAE":
                        x_plot.append(leave_out_year)
                        y_plot.append(np.mean(metrics[name]))
                        y_error.append(2 * np.std(metrics[name]))
                f.write(str(line) + ',')
            f.write('\n')

    # plot values
    x_labels = list(map(lambda x: str(x), x_plot))
    plt.figure()
    plt.errorbar(x_plot, y_plot, yerr=y_error)
    plt.title("MAE vs Year Left Out")
    plt.xlabel("Year Left Out")
    plt.xticks(rotation=45)
    plt.ylabel("Mean Absolute Error")
    plt.xticks(x_plot, x_labels)
    plt.savefig(filename[:-4] + ".png")
