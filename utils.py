import numpy as np
#np.random.seed(11)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import h5py
import os.path as osp
import os
from scipy import ndimage
from glob import glob
from tqdm import tqdm
import sys
'''
Functions used throughout the project.

Set data_root to where your data is saved.
'''

# data_root = '/raid/data/hurricane/'
data_root = 'hurricane_data/'

hand_features = ['vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',
                 'PSLV_v8', 'PSLV_v9', 'PSLV_v10', 'PSLV_v11', 'PSLV_v12', 'PSLV_v13',
                 'PSLV_v14', 'PSLV_v15', 'PSLV_v16', 'PSLV_v17', 'PSLV_v18', 'PSLV_v19',
                 'MTPW_v2', 'MTPW_v3', 'MTPW_v4', 'MTPW_v5', 'MTPW_v6', 'MTPW_v7',
                 'MTPW_v8', 'MTPW_v9', 'MTPW_v10', 'MTPW_v11', 'MTPW_v12', 'MTPW_v13',
                 'MTPW_v14', 'MTPW_v15', 'MTPW_v16', 'MTPW_v17', 'MTPW_v18', 'MTPW_v19',
                 'MTPW_v20', 'MTPW_v21', 'MTPW_v22', 'IR00_v2', 'IR00_v3', 'IR00_v4',
                 'IR00_v5', 'IR00_v6', 'IR00_v7', 'IR00_v8', 'IR00_v9', 'IR00_v10',
                 'IR00_v11', 'IR00_v12', 'IR00_v13', 'IR00_v14', 'IR00_v15', 'IR00_v16',
                 'IR00_v17', 'IR00_v18', 'IR00_v19', 'IR00_v20', 'IR00_v21', 'CSST_t24',
                 'CD20_t24', 'CD26_t24', 'COHC_t24', 'DTL_t24', 'RSST_t24', 'U200_t24',
                 'U20C_t24', 'V20C_t24', 'E000_t24', 'EPOS_t24', 'ENEG_t24', 'EPSS_t24',
                 'ENSS_t24', 'RHLO_t24', 'RHMD_t24', 'RHHI_t24', 'Z850_t24', 'D200_t24',
                 'REFC_t24', 'PEFC_t24', 'T000_t24', 'R000_t24', 'Z000_t24', 'TLAT_t24',
                 'TLON_t24', 'TWAC_t24', 'TWXC_t24', 'G150_t24', 'G200_t24', 'G250_t24',
                 'V000_t24', 'V850_t24', 'V500_t24', 'V300_t24', 'TGRD_t24', 'TADV_t24',
                 'PENC_t24', 'SHDC_t24', 'SDDC_t24', 'SHGC_t24', 'DIVC_t24', 'T150_t24',
                 'T200_t24', 'T250_t24', 'SHRD_t24', 'SHTD_t24', 'SHRS_t24', 'SHTS_t24',
                 'SHRG_t24', 'PENV_t24', 'VMPI_t24', 'VVAV_t24', 'VMFX_t24', 'VVAC_t24',
                 'HE07_t24', 'HE05_t24', 'O500_t24', 'O700_t24', 'CFLX_t24', 'DELV-12']


def load_image(path):
    h5 = h5py.File(path, 'r')
    return h5['matrix'].value


def prepend_subdirs(all_names,names):
    ret = []
    for p in names:
        for q in all_names:
            if p in q:
                ret.append(q)
    return ret


def load_augmented_features():
    x_train = np.load('features_train.npy')
    x_test  = np.load('features_test.npy')
    y_train = np.load('y_train.npy')
    y_test  = np.load('y_test.npy')
    ids = np.load('ids.npy')
    return x_train, x_test, y_train, y_test, ids


def load_image_data_cv():
    # train
    train_df = pd.read_csv(osp.join(data_root, 'gt_64.csv'))
    x_train = np.array([load_image(p) for p in train_df['image_filename'].values])
    y_train = train_df['dv24'].values
    ids = train_df['id'].values
    # test
    x_test, _, y_test = load_augmented_data_cv(test_only=True, image=True)
    # test_df  = pd.read_csv(osp.join(data_root, 'gt_64_2017.csv')
    # y_test  = test_df['dv24'].values
    # x_test = np.array([load_image(p) for p in test_df['image_filename'].values])
    return x_train, x_test, y_train, y_test, ids


def load_hand_data_cv():
    # train
    train_df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv')) #'hand_global_train.csv'
    train_df = train_df.loc[~((train_df.basin=='AL') & (train_df.year==2017))]
    ids = train_df['name'].values
    x_train = np.array(train_df[hand_features].values)
    y_train = train_df[['dvs24']].values
    # test

    test_df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv'))
    test_df = test_df.loc[((test_df.year==2017) & (test_df.type=='opr'))]
    x_test = np.array(test_df[hand_features].values)
    y_test = test_df[['dvs24']].values
    return x_train, x_test, y_train, y_test, ids


def load_augmented_data_cv(test_only=False, image=False):
    if not test_only:
        # train
        train_df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv')) #'NOAA_all_dvs24_vars_w_img_train_clean.csv'
        train_df = train_df.loc[~((train_df.basin=='AL') & (train_df.year==2017))]
        if image:
            train_df = train_df.loc[~train_df.imag_name.isnull()]
        print(('training data size:', train_df.shape))
        y_train_temp = train_df[['dvs24']].values
        ids_temp = train_df['name'].values
        x_train_hand_temp = np.array(train_df[hand_features].values)
        if image:
            print('Loading train images...')
            x_train_images = []
            x_train_hand = []
            y_train = []
            ids = []
            for i,im_name in enumerate(train_df['imag_name'].values):
                #print(im_name,end=', ')
                try:
                    im_path = (glob(osp.join(data_root, f'images_64/*/{im_name}.h5')) + glob(osp.join(data_root, f'images_64_2017/*/{im_name}.h5')))[0]
                    x_train_images.append(load_image(im_path))
                    x_train_hand.append(x_train_hand_temp[i,:])
                    y_train.append(y_train_temp[i])
                    ids.append(ids_temp[i])
                except:
                    pass
            x_train_images = np.array(x_train_images)
            y_train = np.array(y_train)
            x_train_hand = np.array(x_train_hand)
            print(('FINAL training data size:', y_train.shape))

    # test
    test_df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv'))
    test_df = test_df.loc[((test_df.year==2017) & (test_df.type=='opr'))]
    if image:
        test_df = test_df.loc[~test_df.imag_name.isnull()]
    y_test_temp = test_df[['dvs24']].values
    x_test_hand_temp = np.array(test_df[hand_features].values)
    print(('testing size: ', test_df.shape))
    if image:
        print('Loading test images...')
        x_test_images = []
        x_test_hand = []
        y_test = []
        for i,im_name in enumerate(test_df['imag_name'].values):
            #print(im_name, end=', ')
            try:
                im_path = (glob(osp.join(data_root, f'images_64/*/{im_name}.h5')) + glob(osp.join(data_root, f'images_64_2017/*/{im_name}.h5')))[0]
                x_test_images.append(load_image(im_path))
                x_test_hand.append(x_test_hand_temp[i,:])
                y_test.append(y_test_temp[i])
            except:
                pass
        x_test_images = np.array(x_test_images)
        x_test_hand = np.array(x_test_hand)
        y_test = np.array(y_test)
        print(('FINAL testing size: ', y_test.shape))

    if test_only:
        return x_test_images, x_test_hand, y_test
    else:
        return (x_train_images, x_train_hand), (x_test_images, x_test_hand), y_train, y_test, ids


def plot_pred_v_true(y_true, y_pred, **kwargs):
    plt.scatter(y_true, y_pred)
    plt.plot([-100, 100], [-100, 100], '-', color='r')
    plt.xlim(-70,70)
    plt.ylim(-70,70)
    plt.xlabel('True')
    plt.ylabel('Predict')
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'save_path' in kwargs:
        if not osp.exists(osp.dirname(kwargs['save_path'])):
            os.makedirs(osp.dirname(kwargs['save_path']))
        plt.savefig(kwargs['save_path'])
    else:
        plt.show()


def compute_metrics(y_true, y_predict, print_them=False):
    metrics = {'MAE': mean_absolute_error(y_true, y_predict),
               'RMSE': np.sqrt(mean_squared_error(y_true, y_predict)),
               'R^2': r2_score(y_true, y_predict)}
    if print_them:
        for k, v in metrics.items():
            print(f'{k}: {v:.2f}')
        print()
    return metrics


def get_train_hurricane_ids(csv_file=osp.join(data_root, 'train_64.csv')):
    df = pd.read_csv(csv_file)
    return df['id'].values


def image_generator(x, y, batch_sz=32):
    '''
    Data augmentation for cnn_augmented
    '''
    def random_rotate(im):
        theta = np.random.choice([0,90,180,270])
        if theta == 0:
            return im
        else:
            return ndimage.rotate(im, theta)

    x_images = x[0][:]
    x_hand   = x[1][:]
    batches_per_epoch = (len(y) // batch_sz) + 1
    while True:
        # shuffle data sequence
        shuffle = np.random.permutation(len(y))
        x_images = x_images[shuffle]
        x_hand = x_hand[shuffle]
        y = y[shuffle]
        # loop batches
        for b in range(batches_per_epoch):
            x_images_batch = x_images[b*batch_sz:(b+1)*batch_sz]
            x_hand_batch   = x_hand[b*batch_sz:(b+1)*batch_sz]
            x_images_batch = np.array([random_rotate(_) for _ in x_images_batch])
            y_batch = y[b*batch_sz:(b+1)*batch_sz]
            yield [x_images_batch, x_hand_batch], y_batch

def load_loyo_data(leave_out_year, get_hand=False, get_images=False, scale=False, remove_oprreadup=False, remove_oprfortraining=False, data_root=data_root):
    df = pd.read_csv(osp.join(data_root, 'train_global_fill_REA_na_wo_img_scaled_w2020.csv')) #58995 rows
    #df = pd.read_csv(osp.join(data_root, 'train_global_fill_na_w_img_scaled.csv')) # 38k data
    # train
    train_df = df.loc[~((df.basin=='AL') & (df.year==leave_out_year))]
    # if remove duplicated opr and rea training events (the rea part)for AL 2010-2018:
    if remove_oprreadup:
        train_df = train_df.loc[~((train_df.type=='rea') & (train_df.basin=='AL') & (train_df.year>=2010))]
    # remove all opr data points for training:
    if remove_oprfortraining:
        train_df = train_df.loc[~(train_df.type=='opr')]
    ids = train_df['name'].values
    y_train = train_df[['dvs24']].values
    # test
    test_df = df.loc[((df.year==leave_out_year) & (df.type=='opr'))]
    y_test = test_df[['dvs24']].values

    # hand features
    if get_hand:
        x_train_hand = train_df[hand_features].values
        x_test_hand  = test_df[hand_features].values

    # images
    if get_images:
        names_train = train_df['image_name'].values
        names_test  = test_df['image_name'].values

        all_names = [str(p) for p in Path(osp.join(data_root,'image2ch_no_nans_split_64')).rglob('*.h5')]
        paths_train = prepend_subdirs(all_names, names_train)
        paths_test  = prepend_subdirs(all_names, names_test)

        x_train_images = np.array([load_image(p) for p in paths_train])
        x_test_images  = np.array([load_image(p) for p in paths_test])
        if scale:
            means    = [243.78, 1.96]
            std_devs = [30.14, 3.08]
            x_train_images[...,0] = ( x_train_images[...,0] - means[0] ) / std_devs[0]
            x_train_images[...,1] = ( x_train_images[...,1] - means[1] ) / std_devs[1]
            x_test_images[...,0]  = ( x_test_images[...,0] - means[0] ) / std_devs[0]
            x_test_images[...,1]  = ( x_test_images[...,1] - means[1] ) / std_devs[1]

    # returning
    if get_hand and not get_images:
        return x_train_hand, x_test_hand, y_train, y_test, ids
    if get_images and not get_hand:
        return x_train_images, x_test_images, y_train, y_test, ids
    if get_images and get_hand:
        return [x_train_images, x_train_hand], [x_test_images, x_test_hand], y_train, y_test, ids


def get_train_hurricane_names_loyo(leave_out_year,data_root=data_root):
    # load all data into one data frame
    train_df = pd.read_csv(osp.join(data_root, 'train.csv')).append(pd.read_csv(osp.join(data_root, 'test.csv')))

    # take out year from train and take out all except year from test
    df = train_df[train_df.year != leave_out_year]
    df.dropna(axis=0, inplace=True, subset=['DELV-12'])
    return df['name'].values

def predict_with_rotations(model, x, architecture=None):
    '''
    Predict on rotations of the same image
    '''
    y_predict = None
    thetas = (0, 90, 180, 270)
    for theta in thetas:
        if architecture == 'cnn_augmented':
            x_rotated = [np.array([ndimage.rotate(_, theta) for _ in x[0]]), x[1]]
        else:
            x_rotated = np.array([ndimage.rotate(_, theta) for _ in x])
        if y_predict is None:
            y_predict = model.predict(x_rotated)[:,0]
        else:
            y_predict += model.predict(x_rotated)[:,0]
    y_predict /= len(thetas)
    return y_predict

def save_model(model,model_save_filename):
    if model_save_filename:
        print(f'Saving model to {model_save_filename}...')
        model.save(model_save_filename)
    return
