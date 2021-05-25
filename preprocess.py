import os
import h5py
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import os.path as osp
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.switch_backend('agg')
'''
Pre-processing functions.  Most of these were only run once, for example, to load data, modify it, and save it. The 
functions are saved here for possible future use. 
'''


hand_features = ['name','vs0', 'PSLV_v2', 'PSLV_v3', 'PSLV_v4', 'PSLV_v5', 'PSLV_v6', 'PSLV_v7',
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
                 'HE07_t24', 'HE05_t24', 'O500_t24', 'O700_t24', 'CFLX_t24', 'DELV-12',
                 'dvs24']


def path2arr(path):
    h5 = h5py.File(path, 'r')
    return h5['matrix'].value


def get_nan_fracs(arr):
    nan_fracs = []
    for c in range(arr.shape[-1]):
        nan_fracs.append(np.sum(np.isnan(arr[:, :, c])) / np.product(arr[:, :, c].shape))
    return np.array(nan_fracs)


def clean_nans(arr):
    '''
    Input is a 2d array possibly containing nans
    Output is same 2d array with nans replaced by nearest non-nan neighbor
    '''
    def get_neighbor_val(i,j):
        d = 0
        while True:
            d += 1
            for a in range(-d, d+1):
                if i+a < 0: continue  # dont end-index
                for b in range(-d, d+1):
                    if j+b < 0: continue  # dont end-index
                    if abs(a) != d and abs(b) != d: continue  # only iterate over perimeter
                    try:
                        val = arr[i+a,j+b]
                    except IndexError:
                        continue
                    else:
                        if not np.isnan(val):
                            return val

    arr_clean = np.copy(arr)
    for i,j in np.argwhere(np.isnan(arr)):
        arr_clean[i,j] = get_neighbor_val(i,j)
    return arr_clean


def extract_images():
    '''
    Extract images from original big dataset and save separately.
    Remove two middle channels.
    Don't save images with a lot of nans.
    If image has a small number of nans then replace with nearest neighbor values
    '''

    h5_filename = '/raid/data/hurricane/TCIR-ALL_2017.h5'
    output_basepath = '/raid/data/hurricane/images_64_2017'

    h5 = h5py.File(h5_filename, 'r')
    df = pd.read_hdf(h5_filename, key="info", mode='r')
    X = np.array(h5['matrix'])

    output_im_folder = 0
    for i,x in tqdm(enumerate(X), total=len(X)):
        df_row = df.iloc[i]

        # ignore images in 2017 set that are from 2016
        year = df_row['time'][0:4]
        if 'TCIR-ALL_2017.h5' in h5_filename and year == '2016':
            continue

        x_out = x[:,:,[0,-1]].copy()

        nan_fracs = get_nan_fracs(x_out)
        if np.any(nan_fracs > 0.1):
            print(f'Image corrupt... nan fractions: {nan_fracs}... skipping')
            continue
        if nan_fracs[0] > 0.0:
            x_out[:,:,0] = clean_nans(x_out[:,:,0]).copy()
        if nan_fracs[1] > 0.0:
            x_out[:,:,1] = clean_nans(x_out[:,:,1]).copy()

        # standardization
        x_out[:,:,0] -= 185.
        x_out[:,:,0] /= ( 300. - 185. )
        x_out = x_out[68:132,68:132,:]

        if i % 200 == 0:
            output_im_folder += 1
        output_filename = f'{df_row["data_set"]}_{df_row["ID"]}_{df_row["time"]}.h5'
        output_path = osp.join(output_basepath,str(output_im_folder),output_filename)
        if not osp.exists(osp.dirname(output_path)):
            os.makedirs(osp.dirname(output_path))
        with h5py.File(output_path, 'w') as f:
            dset = f.create_dataset('matrix', data=x_out)


def make_gt_df():
    '''
    Make ground truth dataframe.
    For each image that got saved find its 24 our intensity change.
    Save as csv
    '''
    images_root = '/raid/data/hurricane/images_64_2017'
    h5_filename = '/raid/data/hurricane/TCIR-ALL_2017.h5'
    output_filename = '/raid/data/hurricane/gt_64_2017.csv'

    df_all = pd.read_hdf(h5_filename, key="info", mode='r')

    dt = timedelta(hours=24)

    output_dict = {'v0':[],
                   'dv24':[],
                   'id':[],
                   'image_filename':[],
                   'time':[]}

    for im_path in tqdm(glob(osp.join(images_root,'*/*.h5'))):
        im_name = im_path.split('/')[-1]
        id = im_name.split('_')[1]    # e.g. 200801L
        t0_string = im_name.split('_')[-1].replace('.h5','')  # e.g. 2008060115
        t0 = datetime(int(t0_string[0:4]), int(t0_string[4:6]), int(t0_string[6:8]), int(t0_string[8:]))
        t24 = t0 + dt
        t24 = f'{t24.year}{t24.month:02d}{t24.day:02d}{t24.hour:02d}'

        tmp = df_all.loc[(df_all['ID']==id) & (df_all['time']==t24)]
        if tmp.empty:
            # measurement doesn't exist 24 hours later
            continue
        else:
            v24 = tmp['Vmax'].values[0]
            v0 = df_all.loc[(df_all['ID']==id) & (df_all['time']==t0_string)]['Vmax'].values[0]
            dv24 = v24 - v0

        output_dict['v0'].append(v0)
        output_dict['dv24'].append(dv24)
        output_dict['id'].append(id)
        output_dict['image_filename'].append(im_path)
        output_dict['time'].append(t0_string)

    pd.DataFrame(output_dict).to_csv(output_filename, index=False)


def get_pixel_distribution():
    '''
    Analyze distribution of pixel values for each image
    '''
    images_root = '/raid/data/hurricane/images_201'
    for im_path in glob(osp.join(images_root,'*/*.h5')):
        arr = path2arr(im_path)
        print(np.mean(arr[:,:,0]), np.sum(arr[:,:,0]), np.sum(arr[:,:,1]))
        # plt.hist(arr[:,:,0].flatten(),100)
        # plt.savefig('../hist0.png')
        # plt.clf()
        # plt.hist(arr[:,:,1].flatten(),100)
        # plt.savefig('../hist1.png')
        # plt.clf()
        # stats0 = {'mean': np.mean(arr[:,:,0]),
        #           'med':  np.median(arr[:,:,0]),
        #           'std':  np.std(arr[:,:,0])}
        # stats1 = {'mean': np.mean(arr[:,:,1]),
        #           'med':  np.median(arr[:,:,1]),
        #           'std':  np.std(arr[:,:,1])}
        # print(stats0, stats1)
        # print(np.min(arr[:,:,0]), np.max(arr[:,:,0]))
        # input('')


def split_train_test_csv():
    full_csv = '/raid/data/hurricane/gt_64.csv'
    full_df = pd.read_csv(full_csv)
    unique_ids = full_df['id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2)

    train_df = full_df.loc[full_df['id'].isin(train_ids)]
    test_df  = full_df.loc[full_df['id'].isin(test_ids)]

    train_df.to_csv('/raid/data/hurricane/train_64.csv', index=False)
    test_df.to_csv('/raid/data/hurricane/test_64.csv', index=False)


def make_hand_sets():
    df = pd.read_csv('/raid/data/hurricane/hand_global.csv')
    df_train = df.loc[df['year'] != 2017]
    df_test  = df.loc[df['year'] == 2017]
    df_train = df_train[hand_features].dropna()
    df_test  = df_test[hand_features].dropna()
    df_train.to_csv('/raid/data/hurricane/hand_global_train.csv', index=False)
    df_test.to_csv('/raid/data/hurricane/hand_global_test.csv', index=False)


def clean_augmented_df():
    '''
    Remove from the image+hand df any rows for which the image doesn't exist (because it was too corrupt)
    '''
    train_df = pd.read_csv('/raid/data/hurricane/NOAA_all_dvs24_vars_w_img_train.csv')
    test_df  = pd.read_csv('/raid/data/hurricane/NOAA_all_dvs24_vars_w_img_test.csv')
    # Train
    print('--- train ---')
    delete_rows = []
    for i,im_name in enumerate(train_df['imag_name'].values):
        n_matches = len(glob(f'/raid/data/hurricane/images_64/*/{im_name}.h5')) + len(glob(f'/raid/data/hurricane/images_64_2017/*/{im_name}.h5'))
        if n_matches == 0:
            print(f'Missing image {im_name}.  Deleting row...')
            delete_rows.append(i)
    train_df = train_df.drop(delete_rows)

    # Test
    print('--- test ---')
    delete_rows = []
    for i, im_name in enumerate(test_df['imag_name'].values):
        n_matches = len(glob(f'/raid/data/hurricane/images_64/*/{im_name}.h5')) + len(glob(f'/raid/data/hurricane/images_64_2017/*/{im_name}.h5'))
        if n_matches == 0:
            print(f'Missing image {im_name}.  Deleting row...')
            delete_rows.append(i)
    test_df = test_df.drop(delete_rows)

    train_df.to_csv('/raid/data/hurricane/NOAA_all_dvs24_vars_w_img_train_clean.csv')
    test_df.to_csv('/raid/data/hurricane/NOAA_all_dvs24_vars_w_img_test_clean.csv')

globals()[sys.argv[1]]()

