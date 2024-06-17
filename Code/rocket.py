from tsai.all import *
import pandas as pd

def prepare_rocket_data(data, test_data=None, predict=False):

    if predict == False:
        extra_train = pd.DataFrame()
        extra_val = pd.DataFrame()
        extra_test = pd.DataFrame()

        X_train, y_train, X_val, y_val = data
        X_test, y_test = test_data

        if 'sin_time' in X_train.columns:
            extra_train['sin_time'] = X_train['sin_time']
            extra_train['cos_time'] = X_train['cos_time']
            X_train.drop(['sin_time','cos_time'], axis=1, inplace=True)
            extra_val['sin_time'] = X_val['sin_time']
            extra_val['cos_time'] = X_val['cos_time']
            X_val.drop(['sin_time','cos_time'], axis=1, inplace=True)
            extra_test['sin_time'] = X_test['sin_time']
            extra_test['cos_time'] = X_test['cos_time']
            X_test.drop(['sin_time','cos_time'], axis=1, inplace=True)
        if 'sin_martian_time_of_year' in X_train.columns:
            extra_train['sin_martian_time_of_year'] = X_train['sin_martian_time_of_year']
            extra_train['cos_martian_time_of_year'] = X_train['cos_martian_time_of_year']
            X_train.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
            extra_val['sin_martian_time_of_year'] = X_val['sin_martian_time_of_year']
            extra_val['cos_martian_time_of_year'] = X_val['cos_martian_time_of_year']
            X_val.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
            extra_test['sin_martian_time_of_year'] = X_test['sin_martian_time_of_year']
            extra_test['cos_martian_time_of_year'] = X_test['cos_martian_time_of_year']
            X_test.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
        if 'drop width' in X_train.columns:
            extra_train['drop width'] = X_train['drop width']
            X_train.drop('drop width', axis=1, inplace=True)
            extra_val['drop width'] = X_val['drop width']
            X_val.drop('drop width', axis=1, inplace=True)
            extra_test['drop width'] = X_test['drop width']
            X_test.drop('drop width', axis=1, inplace=True)
        if 'max' in X_train.columns:
            extra_train['max'] = X_train['max']
            extra_val['max'] = X_val['max']
            extra_test['max'] = X_test['max']
            X_train.drop('max', axis=1, inplace=True)
            X_val.drop('max', axis=1, inplace=True)
            X_test.drop('max', axis=1, inplace=True)
        if 'max/avg' in X_train.columns:
            extra_train['max/avg'] = X_train['max/avg']
            extra_val['max/avg'] = X_val['max/avg']
            extra_test['max/avg'] = X_test['max/avg']
            X_train.drop('max/avg', axis=1, inplace=True)
            X_val.drop('max/avg', axis=1, inplace=True)
            X_test.drop('max/avg', axis=1, inplace=True)
        if 'time since last pDrop' in X_train.columns:
            extra_train['time since last pDrop'] = X_train['time since last pDrop']
            extra_val['time since last pDrop'] = X_val['time since last pDrop']
            extra_test['time since last pDrop'] = X_test['time since last pDrop']
            X_train.drop('time since last pDrop', axis=1, inplace=True)
            X_val.drop('time since last pDrop', axis=1, inplace=True)
            X_test.drop('time since last pDrop', axis=1, inplace=True)

        data = [X_train, y_train, X_val, y_val]
        test_data =[X_test, y_test]
        extra_features = [extra_train, extra_val, extra_test]
        return data, test_data, extra_features
    
    elif predict == True:

        extra_features = pd.DataFrame()
        if 'sin_time' in data.columns:
            extra = True
            extra_features['sin_time'] = data['sin_time']
            extra_features['cos_time'] = data['cos_time']
            data.drop(['sin_time','cos_time'], axis=1, inplace=True)
        if 'sin_martian_time_of_year' in data.columns:
            extra = True
            extra_features['sin_martian_time_of_year'] = data['sin_martian_time_of_year']
            extra_features['cos_martian_time_of_year'] = data['cos_martian_time_of_year']
            data.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
        if 'drop width' in data.columns:
            extra = True
            extra_features['drop width'] = data['drop width']
            data.drop('drop width', axis=1, inplace=True)
        if 'max' in data.columns:
            extra = True
            extra_features['max'] = data['max']
            data.drop('max', axis=1, inplace=True)
        if 'max/avg' in data.columns:
            extra = True
            extra_features['max/avg'] = data['max/avg']
            data.drop('max/avg', axis=1, inplace=True)
        if 'time since last pDrop' in data.columns:
            extra = True
            extra_features['time since last pDrop'] = data['time since last pDrop']
            data.drop('time since last pDrop', axis=1, inplace=True)
        return data, extra_features

def get_rocket_features(data, window_size, num_features, test_data=None, predict=False, rocket_model=None):
    if predict == False:
        X_train, y_train, X_val, y_val = data
        X_test, y_test = test_data
        X_train = X_train.values.reshape(len(X_train), num_features, window_size)
        X_val = X_val.values.reshape(len(X_val), num_features, window_size)
        X_test = X_test.values.reshape(len(X_test), num_features, window_size)

        tfms = [None, [Categorize()]]
        batch_tfms = [TSStandardize(by_sample=True)]
        dls_train = get_ts_dls(X_train, y_train, splits=None, valid_pct=0.0, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10_000)
        dls_val = get_ts_dls(X_val, y_val, splits=None, valid_pct=0.0, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10_000)
        dls_test = get_ts_dls(X_test, y_test, splits=None, valid_pct=0.0, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10_000)
        model = build_ts_model(ROCKET, dls=dls_train)
        X_train, y_train = create_rocket_features(dls_train.train, model)
        X_val, y_val = create_rocket_features(dls_val.train, model)
        X_test, y_test = create_rocket_features(dls_test.train, model)
        return X_train, X_val, X_test, model,
    
    elif predict == True:
        X = data.values.reshape(len(data), num_features, window_size)
        dummy_y = np.zeros(len(X))
        tfms = [None, [Categorize()]]
        batch_tfms = [TSStandardize(by_sample=True)]
        dls = get_ts_dls(X, y=dummy_y, splits=None, valid_pct=0.0, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10_000)
        X_data, y_data = create_rocket_features(dls.train, rocket_model)
        return X_data
