import numpy as np
import utils
import sys
'''
Baseline models

Usage:

python baselines.py random_forest
or
python baselines.py predict_mean
etc
'''

method = sys.argv[1]

def predict_mean():
    print('Loading data...')
    _, _, y_train, y_test, _ = utils.load_hand_data_cv()
    y_predict = np.mean(y_train)
    utils.compute_metrics(y_test, y_predict, print_them=True)


def no_change():
    # assume v_(t+1) = v_t
    print('Loading data...')
    _, _, _, y_test, _ = utils.load_hand_data_cv()
    y_predict = [0]*len(y_test)
    utils.compute_metrics(y_test, y_predict, print_them=True)


def linear():
    from sklearn.linear_model import LinearRegression
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    model = LinearRegression()
    print('Fitting model...')
    model.fit(x_train, y_train)
    utils.compute_metrics(y_test, model.predict(x_test), print_them=True)


def lasso():
    from sklearn.linear_model import LassoCV
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    print('Fitting model...')
    model = LassoCV(cv=5, random_state=0, max_iter=2000)
    model.fit(x_train, y_train)
    utils.compute_metrics(y_test, model.predict(x_test), print_them=True)


def random_forest():
    from sklearn.ensemble import RandomForestRegressor
    print('Loading data...')
    x_train, x_test, y_train, y_test, _ = utils.load_hand_data_cv()
    print('Fitting model...')
    model = RandomForestRegressor(max_depth=200, random_state=0, n_estimators=100, n_jobs=3, verbose=1)
    model.fit(x_train, y_train)
    utils.compute_metrics(y_test, model.predict(x_test), print_them=True)


locals()[method]()