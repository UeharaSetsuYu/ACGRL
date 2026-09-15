import scipy.io as sio
import numpy as np
from pathlib import Path

from units.unit import normalize


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(dataset_name):
    if dataset_name == 'BBCSport':
        '''
             BBCSportL: Contains 544 samples in 5 clusters for 2 views  3183/3203, 
              '''
        X_list = []
        mat = sio.loadmat(DATA_DIR / 'BBCSport.mat')
        for i in range(2):
            x = mat['X'][0][i].T
            x = x.toarray()

            # print(x.shape)
            # x = normalize(mat['X'][0][i].T, flag="row_vector")
            X_list.append(normalize(x, flag='row_vector'))

        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        return X_list, y

    elif dataset_name == 'NGs':
        '''
            contains 500 instance in 5 clusters for 3 vies and feature dimension is [2000, 2000, 2000]
        '''
        mat = sio.loadmat(DATA_DIR / 'NGs.mat')
        X_list = []
        for view in range(3):
            X = mat['X']
            X_list.append(normalize(X[view][0], flag='row_vector'))
        y = np.squeeze(mat['Y']).astype('int')
        if np.min(y) == 1:
            y = y - 1
        return X_list, y

    elif dataset_name == 'Hdigit':
        '''
            contains 10000 instance in 10 clusters for 2 views and feature dimension is  [784, 256]
        '''
        X_list = []
        mat = sio.loadmat(DATA_DIR / 'Hdigit.mat')
        X = mat['data'][0]
        for view in range(2):
            X_list.append(normalize(X[view].T))
        y = np.squeeze(mat['truelabel'][0][0]).astype('int')
        if np.min(y) == 1:
            y -= 1
        return X_list, y
    elif dataset_name == 'Cora':
        '''contains 2708 instance in 7 clusters for 2 views and feature dimension is  [2708, 1433]'''
        X_list = []
        mat = sio.loadmat(DATA_DIR / 'Cora.mat')
        X1 = mat['coracites']
        X2 = mat['coracontent']
        for i in [X1, X2]:
            X_list.append(normalize(i, 'row_vector'))
        y = np.squeeze(mat['y']).astype('int')
        if np.min(y) == 1:
            y -= 1
        return X_list, y

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
