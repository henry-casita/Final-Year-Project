import helpers
import pandas as pd
from importlib import reload
import random
import models
#import rocket
import numpy as np
import os
from datetime import datetime
from datetime import datetime, timezone
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import joblib

random.seed(42)
np.random.seed(42)


def lstm_training_run(feature_list):

    pressuredrops = pd.read_csv('allPdrops_ordered.csv').sort_values(by=' SOL ')

    unique_sols = pressuredrops[' SOL '].unique()
    missing_sols = []
    for sol in unique_sols:
        file_path = f'seisdata_pq/sol_{sol}_seisdata.parquet'
        if not os.path.isfile(file_path):
            pressuredrops = pressuredrops[pressuredrops[' SOL '] != sol]
            missing_sols.append(sol)

    pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] = pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '].str.strip()
    pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] = pd.to_datetime(pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '], format='%Y-%jT%H:%M:%S.%fZ')
    pressure_threshold = -0

    filtered_drops = pressuredrops[pressuredrops['_DROP_ '] < pressure_threshold]
    window_size = 35
    features = 3
    if 'Gradient' in feature_list:
        features += 1
    if 'Total' in feature_list:
        features += 1
    power_threshold = 1.148e-8
    samples_per_drop = 5
    pos_sample_spacing = 5
    sol_range = [14, 862]
    test_sols = helpers.generate_test_sols(sol_range=sol_range, percentage=0.1)
    pos_df, pDrops_below_thresh, pressuredrops, pDrops_out_of_range, pos_thresh_dict = helpers.construct_pos_dataset(feature_list=feature_list, sol_range=sol_range, test_sols=test_sols, drop_list=filtered_drops, power_threshold=power_threshold, window_size = window_size, samples_per_drop = samples_per_drop, pos_sample_spacing= pos_sample_spacing, verbose=0, drop_magnitude=False)
    pos_length = len(pos_df)
    noise_range = [0, 0]
    neg_df, neg_pDrops_excluded, neg_below_thresh, _ = helpers.construct_neg_samples(feature_list, sol_range, test_sols, pressuredrops, power_threshold, sample_size=window_size, interval=35, verbose=1, noise_range=noise_range)
    neg_length = len(neg_df)

    features_df = pd.concat([pos_df, neg_df], ignore_index=False)
    ratio = len(neg_df) / len(pos_df)
    positive_labels = pd.Series([1] * len(pos_df))
    negative_labels = pd.Series([0] * len(neg_df))
    labels_df = pd.concat([positive_labels, negative_labels], ignore_index=True)
    rows_with_nan = features_df.isnull().any(axis=1)
    indices_with_nan = features_df.index[rows_with_nan]
    labels_df.drop(indices_with_nan, inplace=True)
    features_df.drop(indices_with_nan, inplace=True)
    features_df.columns = features_df.columns.astype(str)
    assert len(features_df) == len(labels_df), f"Mismatch in length of features and labels dataframes, features length {len(features_df)}, labels length {len(labels_df)}"
    feature_scaler = StandardScaler()
    features_scaled = pd.DataFrame(feature_scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)

    joblib.dump(feature_scaler, './lstm_feature_selection/lstm_feature_scaler.joblib')
    print('FEATURES SCALED')
    print(features_scaled.columns)

    if 'Time of day' in feature_list:
        mlst_series = features_scaled.index.to_series().apply(lambda x: helpers.datetime_to_mlst(x))
        features_scaled[['sin_time', 'cos_time']] = mlst_series.apply(helpers.cyclical_encode_time).apply(pd.Series)
    if 'Time of year' in feature_list:
        features_scaled[['sin_martian_time_of_year', 'cos_martian_time_of_year']] = features_scaled.index.to_series().apply(helpers.cyclical_encode_martian_time_of_year).apply(pd.Series)
    if 'Time since last pDrop' in feature_list:
        pressuredrops_time = pressuredrops[[' YYYY-MM-DDTHH:MM:SS.sss ']].sort_values(' YYYY-MM-DDTHH:MM:SS.sss ')
        features_scaled = features_scaled.sort_index()
        features_scaled = pd.merge_asof(features_scaled, pressuredrops_time, left_index=True, right_on=' YYYY-MM-DDTHH:MM:SS.sss ', direction='backward')
        features_scaled['time since last pDrop'] = features_scaled.index - features_scaled[' YYYY-MM-DDTHH:MM:SS.sss ']
        features_scaled['time since last pDrop'] = features_scaled['time since last pDrop'].dt.total_seconds().astype(int)
        features_scaled['time since last pDrop'] = features_scaled['time since last pDrop'].clip(upper=86400)
        features_scaled.drop(' YYYY-MM-DDTHH:MM:SS.sss ', axis=1, inplace=True)

    X_train, X_eval, y_train, y_eval = train_test_split(features_scaled, labels_df, test_size=0.2, random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=123)
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])
    train_data = [X_train, y_train, X_val, y_val]

    adjust_ratio = True
    lstm_eval_model, lstm_final_model = models.train_lstm_test(data=[df.copy() for df in train_data], test_data = [X_test.copy(), y_test.copy()], window_size=window_size, num_features=features, gridsearch=True, verbose=0, ratio=ratio, adjust_ratio=adjust_ratio)
    y_pred = (models.predict_lstm(lstm_eval_model, X_test.copy(), window_size, features) > 0.5).astype("int32")
    class_report_lstm = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", class_report_lstm)

    save = True
    eval_mode = 'test_eval'
    sample_interval = 1
    if eval_mode == 'train_eval':
        eval_sols = [num for num in range(sol_range[0], sol_range[1]) if num not in test_sols]
    elif eval_mode == 'test_eval':
        eval_sols = test_sols
    reload(helpers)
    model_type = 'lstm'
    if model_type == 'xgb': 
        model = best_xgb_clf
        class_report = class_report_xgb
        #model = xgb.XGBClassifier()
        #model.load_model('xgboost_model.bin')
    elif model_type == 'lstm':
        model = lstm_final_model
        class_report = class_report_lstm
    elif model_type == 'xgb_rocket':
        model = best_xgb_rocket_clf
    elif model_type == 'classic rocket':
        model = ridge
    elif model_type == 'logistic rocket':
        model = logistic_clf
        class_report = class_report_rocket
    raw_predictions, total_pdrops_excluded, total_below_thresh = helpers.predict(feature_list=feature_list, test_sols=eval_sols, best_clf=model, model_type=model_type, feature_scaler=feature_scaler, pressuredrops=pressuredrops, power_threshold=power_threshold, sample_interval=sample_interval, sample_size=window_size, features=features)
    best_F1 = 0
    best_match_threshold = 0
    for match_threshold in range(0,27):
        result = helpers.evaluate_predictions(raw_predictions, eval_sols, pressuredrops, match_threshold=match_threshold, sample_size=window_size)
        F1 = result['f1_score']
        if F1 > best_F1:
            best_F1 = F1
            eval_result = result
            best_match_threshold = match_threshold
    positive_predictions = eval_result['predictions']

    if save == True:
        #directory_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}_{model_type}_{pressure_threshold}_{power_threshold}"
        feature_names = '_'.join(feature_list)
        directory_name = f"./lstm_feature_selection/{model_type}_{pressure_threshold}_{power_threshold}_{feature_names}"
        log_string = helpers.log_results(pressure_threshold, power_threshold, window_size, samples_per_drop, pos_sample_spacing, sol_range, pos_length, len(pDrops_below_thresh), neg_length, neg_below_thresh, adjust_ratio, model, model_type, class_report, eval_mode, sample_interval, best_match_threshold, eval_result, feature_list)
        helpers.save(raw_predictions, positive_predictions, log_string, model, directory_name)

    return eval_result['f1_score']


empty_feature_list = ['Time of day', 'Max vs average']
f1_reference = lstm_training_run(empty_feature_list)

feature_list = [
    'Total',
    'Time of day',
    'Time of year',
    'Time since last pDrop',
    'Drop width',
    'Max power',
    'Max vs average',
    'Gradient'
]

chosen_features = []
candidate_features = feature_list.copy()
best_f1 = f1_reference
f1_scores = {}
while True:
    best_feature = None
    f1_reference = best_f1
    features_to_remove = []
    for feature in candidate_features:
        
        current_features = chosen_features + [feature]
        f1 = lstm_training_run(current_features)
        
        feature_combination = ', '.join(current_features)
        print("Feature combination", feature_combination)
        f1_scores[feature_combination] = f1
        
        if f1 > best_f1:
            best_feature = feature
            best_f1 = f1
        if f1 < f1_reference:
            features_to_remove.append(feature)

    if best_feature is None:
        break

    chosen_features.append(best_feature)
    candidate_features.remove(best_feature)
    for feature in features_to_remove:
        if feature in candidate_features:
            candidate_features.remove(feature)


print("Chosen features:", chosen_features)
print("Final F1 score:", f1_reference)

sorted_f1_scores = dict(sorted(f1_scores.items(), key=lambda x: x[1], reverse=True))

with open('lstm_f1_scores.json', 'w') as file:
    json.dump(sorted_f1_scores, file, indent=4)