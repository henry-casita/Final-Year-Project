import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import clone_model
import copy
from sklearn.linear_model import LogisticRegression
import random
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
import torch.nn as nn
import torch

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


class XGBClassifierWithEarlyStopping(xgb.XGBClassifier):
    def __init__(self, early_stopping_rounds=100, **kwargs):
        super().__init__(**kwargs)
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, **kwargs):
        eval_set = [(X, y)]
        if 'eval_set' in kwargs:
            eval_set = kwargs.pop('eval_set')
        self.set_params(early_stopping_rounds=self.early_stopping_rounds)
        return super().fit(X, y, eval_set=eval_set, verbose=False, **kwargs)
    
class XGBRegressorWithEarlyStopping(xgb.XGBRegressor):
    def __init__(self, early_stopping_rounds=100, **kwargs):
        super().__init__(**kwargs)
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, **kwargs):
        eval_set = [(X, y)]
        if 'eval_set' in kwargs:
            eval_set = kwargs.pop('eval_set')
        self.set_params(early_stopping_rounds=self.early_stopping_rounds)
        return super().fit(X, y, eval_set=eval_set, verbose=False, **kwargs)



def train_xgboost_regressor(data=None, bayes_search=False, verbose=0, ratio=1, adjust_ratio=False):

    X_train, y_train, X_val, y_val = data

    if adjust_ratio:
        ratio = ratio
    else:
        ratio = 1

    if bayes_search:
        param_space = {
            'max_depth': Integer(2, 6),
            'learning_rate': Real(0.01, 0.1),
            'alpha': Integer(5, 25),
            'device': ['cuda'],
            # 'scale_pos_weight': [ratio]
        }
        n_iter = 50
    else:
        param_space = {
            'max_depth': [4],
            'learning_rate': [0.1],
            'alpha': [10],
            'device': ['cuda'],
            # 'scale_pos_weight': [ratio]
        }
        n_iter = 1

    xgb_reg_custom = XGBRegressorWithEarlyStopping(objective='reg:squarederror', n_estimators=5000)

    bayes_search = BayesSearchCV(
        estimator=xgb_reg_custom,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=3,
        verbose=verbose,
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best score found: ", bayes_search.best_score_)

    best_model = bayes_search.best_estimator_

    best_model_copy = copy.deepcopy(best_model)

    return best_model_copy

def train_xgboost(data=None, bayes_search=False, verbose=0, ratio=1, adjust_ratio=False):

    X_train, y_train, X_val, y_val = data

    if adjust_ratio:
        ratio = ratio
    else:
        ratio = 1

    if bayes_search:
        param_space = {
            'max_depth': Integer(2, 6),
            'learning_rate': Real(0.01, 0.1),
            'alpha': Integer(5, 25),
            'device': ['cuda'],
            'scale_pos_weight': [ratio]
        }
        n_iter = 50
    else:
        param_space = {
            'max_depth': [6],
            'learning_rate': [0.023383044751314039],
            'alpha': [5],
            'device': ['cuda'],
            'scale_pos_weight': [ratio]
        }
        n_iter = 1

    xgb_clf_custom = XGBClassifierWithEarlyStopping(objective='binary:logistic', n_estimators=5000)

    bayes_search = BayesSearchCV(
        estimator=xgb_clf_custom,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=3,
        verbose=verbose,
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best score found: ", bayes_search.best_score_)

    best_model = bayes_search.best_estimator_
    best_model_copy = copy.deepcopy(best_model)

    return best_model_copy


def train_xgboost_rocket(data=None, bayes_search=False, verbose=0, ratio=1, adjust_ratio=False):

    if data is not None:
        X_train, y_train, X_val, y_val = data
    else:
        X_train = pd.read_parquet('./rocket_datasets/X_train.parquet').to_numpy(copy=False)
        X_val = pd.read_parquet('./rocket_datasets/X_val.parquet').to_numpy(copy=False)
        y_train = pd.read_parquet('./rocket_datasets/y_train.parquet').to_numpy(copy=False)
        y_val = pd.read_parquet('./rocket_datasets/y_val.parquet').to_numpy(copy=False)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        del X_train, X_val, y_train, y_val



    if adjust_ratio:
        ratio = ratio
    else:
        ratio = 1
    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    best_model_copy = copy.deepcopy(xgb_model)

    return best_model_copy


def train_lstm(data, test_data, window_size, num_features, gridsearch=False, verbose=0, ratio=1, adjust_ratio=False):

    print(tf.config.list_physical_devices('GPU'))

    X_train, y_train, X_val, y_val = data
    X_test, y_test = test_data
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    X_train = X_train.values.reshape((len(X_train), window_size, num_features))
    X_val = X_val.values.reshape((len(X_val), window_size, num_features))
    X_test = X_test.values.reshape((len(X_test), window_size, num_features))
    X_train_final = X_train_final.values.reshape((len(X_train_final), window_size, num_features))

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(units=hp.Choice('units', values=[32, 64, 128, 256, 512]),
                    activation='tanh', recurrent_activation='sigmoid', 
                    input_shape=(window_size, num_features)))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        return model
    
    if gridsearch: 
        max_trials = 10
    else: 
        max_trials = 1

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=3,
        directory='my_dir',
        project_name='hparam_tuning')

    stop_early = EarlyStopping(monitor='val_accuracy', patience=50, verbose=verbose, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[stop_early], verbose=verbose)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), verbose=verbose, callbacks=[stop_early])

    best_model_copy = clone_model(best_model)
    best_model_copy.set_weights(best_model.get_weights())

    final_model = tuner.hypermodel.build(best_hps)
    final_model.fit(X_train_final, y_train_final, epochs=500, validation_data=(X_test, y_test), verbose=verbose, callbacks=[stop_early])

    final_model_copy = clone_model(final_model)
    final_model_copy.set_weights(final_model.get_weights())

    return best_model_copy, final_model_copy

def predict_lstm(model, data, window_size, num_features):
    
    extra = False
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

    X = data.values.reshape((len(data), window_size, num_features))
    
    if extra == True:
        predictions = model.predict([X, extra_features])
    else:
        predictions = model.predict(X)
    
    return (predictions > 0.5).astype("int32")


def train_lstm_test(data, test_data, window_size, num_features, gridsearch=False, verbose=0, ratio=1, adjust_ratio=False):

    num_extra_features = 0
    extra_features = False
    extra_train = pd.DataFrame()
    extra_val = pd.DataFrame()
    extra_test = pd.DataFrame()

    print(tf.config.list_physical_devices('GPU'))

    X_train, y_train, X_val, y_val = data
    X_test, y_test = test_data

    if 'sin_time' in X_train.columns:
        extra_features = True
        extra_train['sin_time'] = X_train['sin_time']
        extra_train['cos_time'] = X_train['cos_time']
        X_train.drop(['sin_time','cos_time'], axis=1, inplace=True)
        extra_val['sin_time'] = X_val['sin_time']
        extra_val['cos_time'] = X_val['cos_time']
        X_val.drop(['sin_time','cos_time'], axis=1, inplace=True)
        extra_test['sin_time'] = X_test['sin_time']
        extra_test['cos_time'] = X_test['cos_time']
        X_test.drop(['sin_time','cos_time'], axis=1, inplace=True)
        num_extra_features += 2
    if 'sin_martian_time_of_year' in X_train.columns:
        extra_features = True
        extra_train['sin_martian_time_of_year'] = X_train['sin_martian_time_of_year']
        extra_train['cos_martian_time_of_year'] = X_train['cos_martian_time_of_year']
        X_train.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
        extra_val['sin_martian_time_of_year'] = X_val['sin_martian_time_of_year']
        extra_val['cos_martian_time_of_year'] = X_val['cos_martian_time_of_year']
        X_val.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
        extra_test['sin_martian_time_of_year'] = X_test['sin_martian_time_of_year']
        extra_test['cos_martian_time_of_year'] = X_test['cos_martian_time_of_year']
        X_test.drop(['sin_martian_time_of_year','cos_martian_time_of_year'], axis=1, inplace=True)
        num_extra_features += 2   
    if 'drop width' in X_train.columns:
        extra_features = True
        extra_train['drop width'] = X_train['drop width']
        X_train.drop('drop width', axis=1, inplace=True)
        extra_val['drop width'] = X_val['drop width']
        X_val.drop('drop width', axis=1, inplace=True)
        extra_test['drop width'] = X_test['drop width']
        X_test.drop('drop width', axis=1, inplace=True)
        num_extra_features += 1
    if 'max' in X_train.columns:
        extra_features = True
        extra_train['max'] = X_train['max']
        extra_val['max'] = X_val['max']
        extra_test['max'] = X_test['max']
        X_train.drop('max', axis=1, inplace=True)
        X_val.drop('max', axis=1, inplace=True)
        X_test.drop('max', axis=1, inplace=True)
        num_extra_features += 1
    if 'max/avg' in X_train.columns:
        extra_features = True
        extra_train['max/avg'] = X_train['max/avg']
        extra_val['max/avg'] = X_val['max/avg']
        extra_test['max/avg'] = X_test['max/avg']
        X_train.drop('max/avg', axis=1, inplace=True)
        X_val.drop('max/avg', axis=1, inplace=True)
        X_test.drop('max/avg', axis=1, inplace=True)
        num_extra_features += 1
    if 'time since last pDrop' in X_train.columns:
        extra_features = True
        extra_train['time since last pDrop'] = X_train['time since last pDrop']
        extra_val['time since last pDrop'] = X_val['time since last pDrop']
        extra_test['time since last pDrop'] = X_test['time since last pDrop']
        X_train.drop('time since last pDrop', axis=1, inplace=True)
        X_val.drop('time since last pDrop', axis=1, inplace=True)
        X_test.drop('time since last pDrop', axis=1, inplace=True)
        num_extra_features += 1

    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])  
    X_train = X_train.values.reshape((len(X_train), window_size, num_features))
    X_val = X_val.values.reshape((len(X_val), window_size, num_features))
    X_test = X_test.values.reshape((len(X_test), window_size, num_features))
    X_train_final = X_train_final.values.reshape((len(X_train_final), window_size, num_features))

    if extra_features == True:
        extra_train_final = pd.concat([extra_train, extra_val])

    def build_model(hp):
        time_series_input = Input(shape=(window_size, num_features))
        lstm_output = LSTM(units=hp.Choice('units', values=[32, 64, 128, 256, 512]),
                           activation='tanh', recurrent_activation='sigmoid')(time_series_input)
        
        if extra_features == True:
            extra_input = Input(shape=(num_extra_features,))
            concatenated = Concatenate()([lstm_output, extra_input])
            dropout = Dropout(rate=hp.Float('dropout', min_value=0, max_value=0.5, step=0.1))(concatenated)
        else:
            dropout = Dropout(rate=hp.Float('dropout', min_value=0, max_value=0.5, step=0.1))(lstm_output)
        
        output = Dense(1, activation='sigmoid')(dropout)
        
        if extra_features == True:
            print('Model build test')
            model = Model(inputs=[time_series_input, extra_input], outputs=output)
        else:
            model = Model(inputs=time_series_input, outputs=output)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model

    if gridsearch: 
        max_trials = 10
    else: 
        max_trials = 1

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=3,
        directory='my_dir',
        project_name='hparam_tuning')

    stop_early = EarlyStopping(monitor='val_accuracy', patience=50, verbose=verbose, restore_best_weights=True)
    
    
    if extra_features == True:
        print('test')
        tuner.search([X_train, extra_train], y_train, epochs=500, validation_data=([X_val, extra_val], y_val), callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit([X_train, extra_train], y_train, epochs=500, validation_data=([X_val, extra_val], y_val), verbose=verbose, callbacks=[stop_early])
        best_model_copy = clone_model(best_model)
        best_model_copy.set_weights(best_model.get_weights())
        final_model = tuner.hypermodel.build(best_hps)
        final_model.fit([X_train_final, extra_train_final], y_train_final, epochs=500, validation_data=([X_test, extra_test], y_test), verbose=verbose, callbacks=[stop_early])
        final_model_copy = clone_model(final_model)
        final_model_copy.set_weights(final_model.get_weights())

    else:
        tuner.search(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), verbose=verbose, callbacks=[stop_early])
        best_model_copy = clone_model(best_model)
        best_model_copy.set_weights(best_model.get_weights())
        final_model = tuner.hypermodel.build(best_hps)
        final_model.fit(X_train_final, y_train_final, epochs=500, validation_data=(X_test, y_test), verbose=verbose, callbacks=[stop_early])
        final_model_copy = clone_model(final_model)
        final_model_copy.set_weights(final_model.get_weights())
    
    
    return best_model_copy, final_model_copy




def train_logistic_regression(data, search_best_params=True, verbose=0):

    X_train, y_train, X_test, y_test = data

    if search_best_params:
        eps = 1e-6
        Cs = [7.5e-5, 1e-4]

        best_loss = np.inf
        best_eps = None
        best_C = None
        best_train_score = None
        best_val_score = None
        
        for i, C in enumerate(Cs):
            if verbose > 0:
                print("Iteration ", i)
            
            classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1, verbose=0, max_iter=10000, class_weight='balanced')
            classifier.fit(X_train, y_train)
            
            probas = classifier.predict_proba(X_train)
            loss = nn.CrossEntropyLoss()(torch.tensor(probas), torch.tensor(y_train)).item()
            
            train_score = classifier.score(X_train, y_train)
            val_score = classifier.score(X_test, y_test)
            
            if loss < best_loss:
                best_eps = eps
                best_C = C
                best_loss = loss
                best_train_score = train_score
                best_val_score = val_score
            if verbose > 0: 
                print('{:2} eps: {:.2E} C: {:.2E} loss: {:.5f} train_acc: {:.5f} valid_acc: {:.5f}'.format(
                    i, eps, C, loss, train_score, val_score))
        
        print('\nBest result:')
        print('eps: {:.2E} C: {:.2E} train_loss: {:.5f} train_acc: {:.5f} valid_acc: {:.5f}'.format(
            best_eps, best_C, best_loss, best_train_score, best_val_score))
        
        best_classifier = LogisticRegression(penalty='l2', C=best_C, n_jobs=-1, verbose=0, max_iter=10000, class_weight='balanced')
        best_classifier.fit(X_train, y_train)
        
        return best_classifier
    
    else:

        eps = 1e-6
        C = 7.5e-5
        
        classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1, verbose=0, max_iter = 10000, class_weight = 'balanced')
        classifier.fit(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))
        
        return classifier



