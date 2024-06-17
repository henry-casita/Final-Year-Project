from datetime import datetime, timedelta
import pandas as pd
import random
import math
import numpy as np
import os
import json
import time
from intervaltree import Interval, IntervalTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from skyfield.api import load
from datetime import datetime
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import rocket
import models
from importlib import reload
reload(rocket)

random.seed(42)
np.random.seed(42)

def cyclical_encode_martian_time_of_year(date):

    reference_date = datetime(2019, 3, 23)
    martian_year_length = 668.6
    days_since_reference = (date - reference_date).days
    sol_of_year = (days_since_reference % martian_year_length) / martian_year_length
    sin_component = np.sin(2 * np.pi * sol_of_year)
    cos_component = np.cos(2 * np.pi * sol_of_year)

    return sin_component, cos_component

def cyclical_encode_time(time_value):
    
    normalized_time = time_value / 24
    sin_component = np.sin(2 * np.pi * normalized_time)
    cos_component = np.cos(2 * np.pi * normalized_time)

    return sin_component, cos_component

def datetime_to_mlst(dt):

    marsTime = utctoLTST(dt, 224.376553)
    LTST = marsTime[1] + marsTime[2]/60 + marsTime[3]/3600

    return LTST



def plot_interactive_peak_to_peak_diff_histogram(df):

    peak_to_peak_diff = np.ptp(df, axis=1)
    
    fig = make_subplots(rows=1, cols=1, subplot_titles=("CDF of Peak-to-Peak Difference"))

    sorted_diff = np.sort(peak_to_peak_diff)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    fig.add_trace(go.Scatter(x=sorted_diff, y=cdf, mode='lines', line=dict(color='#ff7f0e', width=2)), row=1, col=1)

    fig.update_layout(
        title="Peak-to-Peak Difference Analysis",
        xaxis=dict(
            title="Peak-to-Peak Difference",
            type="log",
            range=[np.log10(sorted_diff.min()), np.log10(1e-6)],
            tickformat="e"
        ),
        yaxis=dict(title="Cumulative Probability"),
        autosize=True,  
        showlegend=False
    )
    
    pio.show(fig, renderer='browser')


def log_results(pressure_threshold, power_threshold, window_size, samples_per_drop, pos_sample_spacing, sol_range, pos_length, pDrops_below_thresh, neg_length, neg_below_thresh, adjust_ratio, model, model_type, class_report, eval_mode, sample_interval, match_threshold, eval_result, feature_list):
    if model_type == 'xgb':
        model_params = model.get_params()
        non_null_params = {param: value for param, value in model_params.items() if value is not None}
        params_json = json.dumps(non_null_params, indent=4)
    elif model_type == 'lstm':
        model_params = model.get_config()
        non_null_params = {param: value for param, value in model_params.items() if value is not None}
        params_json = json.dumps(non_null_params, indent=4)
    elif model_type == 'logistic rocket':
        model_params = model.get_params()
        converted_params = {}
        for param, value in model_params.items():
            if isinstance(param, np.int64) or isinstance(param, np.float64) or isinstance(param, int) or isinstance(param, float):
                param = str(param)
            
            if isinstance(value, np.int64) or isinstance(value, np.float64) or isinstance(value, int) or isinstance(value, float):
                value = str(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                value = [str(v) for v in value]
            elif isinstance(value, dict):
                value = {str(k): str(v) for k, v in value.items()}
        non_null_params = {str(param): value for param, value in converted_params.items() if value is not None}
        params_json = json.dumps(non_null_params, indent=4)

    log_string = f"""
    Features: {', '.join(feature_list)}

    Preprocessing:
    Sol range: {sol_range},
    Pressure threshold: {pressure_threshold}, Power threshold: {power_threshold},
    Sample size: {window_size}, Samples per drop: {samples_per_drop}, Positive sample spacing: {pos_sample_spacing}
    Number of positive samples: {pos_length}, number of pDrops below threshold: {pDrops_below_thresh}, {pDrops_below_thresh/(pos_length+pDrops_below_thresh)}%
    Number of negative samples: {neg_length}, number of negative samples below threshold: {neg_below_thresh}, {neg_below_thresh/(neg_below_thresh+neg_length)}%

    Classifier Training:
    Classifier: {model_type},
    Params: {params_json}
    Adjust ratio: {adjust_ratio}
    Classification Report:
    {class_report}

    Evaluation:
    Eval mode: {eval_mode}, Sampling interval: {sample_interval}, Match threshold: {match_threshold}

    Evaluation Metrics:
    True Positives (TP): {eval_result['TP']}
    False Negatives (FN): {eval_result['FN']}
    False Positives (FP): {eval_result['FP']}
    Unique Events Matched: {eval_result['unique_events_matched']}
    Precision: {eval_result['precision']:.4f}
    Recall: {eval_result['recall']:.4f}
    F1 Score: {eval_result['f1_score']:.4f}
    """
    return log_string

def save(raw_predictions, positive_predictions, log_string, model, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    log_file = os.path.join(directory, "log.txt")
    with open(log_file, "w") as file:
        file.write(log_string)

    raw_predictions_file = os.path.join(directory, "raw_predictions.json")
    for sol, predictions in raw_predictions.items():
        raw_predictions[sol] = [
            [
                timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(timestamp, datetime) else timestamp,
            ]
            for timestamp in predictions
        ]
    with open(raw_predictions_file, "w") as file:
        json.dump(raw_predictions, file)

    positive_predictions_file = os.path.join(directory, "positive_predictions.json")
    for sol, predictions in positive_predictions.items():
        positive_predictions[sol] = [
            [
                timestamp1.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(timestamp1, datetime) else timestamp1,
                timestamp2.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(timestamp2, datetime) else timestamp2
            ]
            for timestamp1, timestamp2 in predictions
        ]
    with open(positive_predictions_file, "w") as file:
        json.dump(positive_predictions, file)

    model_file = os.path.join(directory, "model.joblib")
    joblib.dump(model, model_file)


def combine_csv_to_parquet(source_folder, output_file):

    dataframes = []

    for file in os.listdir(source_folder):
        if file.endswith(".csv"):

            sol_number = file.split('_')[1]

            print("Processing sol: ", sol_number)
            file_path = os.path.join(source_folder, file)
            #df = pd.read_parquet(file_path)
            #df.set_index('Time', inplace=True)
            #df.index = pd.to_datetime(df.index)
            df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time', dtype='float64')
            df.index = pd.to_datetime(df.index)
            df['sol_number'] = sol_number
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes)
    combined_df.sort_index(inplace=True)
    combined_df.to_parquet(output_file, partition_cols=['sol_number'])

    print(f"Combined Parquet file created at {output_file}")


def convert_csv_to_parquet(source_folder, destination_folder):

    os.makedirs(destination_folder, exist_ok=True)
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(source_folder, csv_file)
        df = pd.read_csv(csv_path)
        parquet_path = os.path.join(destination_folder, csv_file.replace('.csv', '.parquet'))
        df.to_parquet(parquet_path, index=False)
        print(f"Converted {csv_file} to Parquet format.")

def create_sample(feature_list, df_slice, cols_to_exclude):

    if 'Drop width' in feature_list:
        lowest_indices = np.argsort(df_slice['Total'].values)[:2]
        drop_width = np.abs(lowest_indices[1] - lowest_indices[0])
    if 'Max power' in feature_list:
        max = np.max(df_slice['Total'])
    if 'Max vs average' in feature_list:
        ratio = np.max(df_slice['Total'])/np.mean(df_slice['Total'])
    if 'Gradient' in feature_list:
        df_slice.loc[:, 'Gradient'] = df_slice['Total'].diff().ffill().bfill()

    flattened_data = df_slice.drop(columns=cols_to_exclude).values.ravel(order='C') 
    flattened_slice = pd.DataFrame([flattened_data])
    flattened_slice.reset_index(drop=True, inplace=True)
    flattened_slice.columns = range(flattened_slice.shape[1])

    if 'Drop width' in feature_list:
        flattened_slice['drop width'] = drop_width
    if 'Max power' in feature_list:
        flattened_slice['max'] = max
    if 'Max vs average' in feature_list:
        flattened_slice['max/avg'] = ratio

    flattened_slice.index = [df_slice.index[0]]
    return flattened_slice


def generate_test_sols(sol_range, percentage):
    start, end = sol_range
    missing_sols = []
    available_sols = []

    for sol in range(start, end + 1):
        file_name = f"sol_{sol}_seisdata.parquet"
        file_path = os.path.join("seisdata_pq", file_name)
        
        if os.path.exists(file_path):
            available_sols.append(sol)

    num_test_sols = int(len(available_sols) * percentage)
    test_sols = random.sample(available_sols, num_test_sols)
    
    return test_sols


def convert_to_datetime(timestamp):

    timestamp = timestamp.strip()
    if timestamp.endswith('Z'):
        timestamp = timestamp[:-1]
    date_str, time_str = timestamp.split('T')
    year = int(date_str[:4])
    julian_day = int(date_str[5:8])
    dt = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
    full_datetime = datetime.strptime(f'{dt.date()} {time_str}', '%Y-%m-%d %H:%M:%S.%f')

    return full_datetime

def filter_samples(df, threshold):

    squared_df = df.applymap(lambda x: np.square(x))
    squared_df_1 = squared_df.iloc[:, 0::3].set_axis(range(35), axis=1)
    squared_df_2 = squared_df.iloc[:, 1::3].set_axis(range(35), axis=1)
    squared_df_3 = squared_df.iloc[:, 2::3].set_axis(range(35), axis=1)
    summed_df = squared_df_1 + squared_df_2 + squared_df_3
    sqrt_df = summed_df.applymap(lambda x: np.sqrt(x))
    peak_to_peak_diff = sqrt_df.apply(lambda x: np.ptp(x), axis=1)
    keep_mask = peak_to_peak_diff > threshold
    filtered_df = df[keep_mask]
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

def construct_samples(feature_list, sol_number, sol_data, pressuredrops, power_threshold, midfreq_data, highfreq_data, mode='all', sample_size=35, interval=21, verbose=0, noise_range = None):

    out_of_range = 0
    pdrops_excluded = set()
    below_thresh = 0
    time_step= pd.Timedelta(seconds=2.75)
    midfreq_df = pd.DataFrame()
    highfreq_df = pd.DataFrame()
    noise_df = pd.DataFrame()
    try:
        seisdata = sol_data

        samples_df = pd.DataFrame()
        sol_pressuredrops = pressuredrops[pressuredrops[' SOL '] == sol_number]

        for start in range(0, len(seisdata)-sample_size+interval, interval):

            if start + sample_size > len(seisdata):
                start = len(seisdata) - sample_size

            sample = seisdata.iloc[start:start + sample_size]
            if midfreq_data is not None:
                midfreq_slice = midfreq_data.iloc[start:start + sample_size]
            if highfreq_data is not None:
                highfreq_slice = highfreq_data.iloc[start:start + sample_size]

            if mode == 'neg':
                offset = (sample_size/2)*time_step
                drops_overlapping = [x for x in sol_pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] if (x - offset <= sample.index[0] <= x + offset) or (x - offset <= sample.index[-1] <= x + offset)]
                if len(drops_overlapping) > 0:
                    pdrops_excluded.update(drops_overlapping)
                    continue

            pressuredrops_in_sample = sol_pressuredrops[(sol_pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] >= sample.index[0]) & (sol_pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] <= sample.index[-1])]

            if np.ptp(sample['Total']) > power_threshold:
                if 'Total' in feature_list:
                    cols_to_exclude = ['sol_number']
                else:
                    cols_to_exclude = ['Total','sol_number']
                samples_df = samples_df._append(create_sample(feature_list, sample.copy(), cols_to_exclude=cols_to_exclude))
                if midfreq_data is not None:
                    midfreq_df = midfreq_df._append(create_sample(feature_list, midfreq_slice.copy(), cols_to_exclude=cols_to_exclude))
                if highfreq_data is not None:
                    highfreq_df = highfreq_df._append(create_sample(feature_list, highfreq_slice.copy(), cols_to_exclude=cols_to_exclude))

            else:
                below_thresh +=1
                if len(pressuredrops_in_sample) > 0:
                    if mode == 'neg':
                        raise RuntimeError("Pressuredrops in sample which should be negative")
                    pdrops_excluded.update(pressuredrops_in_sample[' YYYY-MM-DDTHH:MM:SS.sss '])
            
            if noise_range is not None:
                if np.ptp(sample['Total']) > noise_range[0] and np.ptp(sample['Total']) < noise_range[1]:
                    sample = sample.copy()
                    total_mean = sample['Total'].mean()
                    sample.loc[:, 'Total'] = sample['Total'] - total_mean
                    z_mean = sample['Z'].mean()
                    sample.loc[:, 'Z'] = sample['Z'] - z_mean
                    e_mean = sample['E'].mean()
                    sample.loc[:, 'E'] = sample['E'] - e_mean
                    n_mean = sample['N'].mean()
                    sample.loc[:, 'N'] = sample['N'] - n_mean
                    if 'Total' in feature_list:
                        cols_to_exclude = ['sol_number']
                    else:
                        cols_to_exclude = ['Total','sol_number']
                    noise_df = noise_df._append(create_sample(feature_list, sample.copy(), cols_to_exclude=cols_to_exclude))

    except Exception as e:
        if verbose > 0:
            print(f"{e}")
        return None, 0
    
    if midfreq_data is not None:
        midfreq_df.columns = range(samples_df.shape[1],samples_df.shape[1]+midfreq_df.shape[1])
        samples_df = pd.concat([samples_df, midfreq_df], axis=1)
    if highfreq_data is not None:
        highfreq_df.columns = range(samples_df.shape[1],samples_df.shape[1]+highfreq_df.shape[1])
        samples_df = pd.concat([samples_df, highfreq_df], axis=1)
    
    return samples_df, pdrops_excluded, below_thresh, noise_df

def construct_pos_dataset(feature_list, sol_range, test_sols, drop_list, power_threshold, window_size = 19, samples_per_drop = 1, pos_sample_spacing = 4, verbose=0, drop_magnitude=False):
    pos_df = pd.DataFrame()
    pDrops_below_thresh = set()
    pDrops_out_of_range = set()
    drop_strengths = []
    file_path = './seisdata.parquet'
    midfreqdata = './seisdata_1.1Hz-3.9Hz.parquet'
    highfreqdata = 'seisdata_4.1Hz-8Hz.parquet'
    if 'Mid frequency data' in feature_list:
        midfreq_df = pd.read_parquet(midfreqdata)
        pos_midfreq = pd.DataFrame()
    if 'High frequency data' in feature_list:
        highfreq_df = pd.read_parquet(highfreqdata)
        pos_highfreq = pd.DataFrame()
    df = pd.read_parquet(file_path)
    thresh_dict = {}

    for sol_number in range(sol_range[0], sol_range[1]):
        excluded = set()
        rows_to_delete = []
        if sol_number not in test_sols:
            try:
                if verbose > 0:
                    print(f"Processing SOL {sol_number}")

                start_time = time.time()
                seisdata = df[df['sol_number'] == sol_number]
                if 'Mid frequency data' in feature_list:
                    midfreq_data = midfreq_df[midfreq_df['sol_number'] == sol_number]
                if 'High frequency data' in feature_list:
                    highfreq_data = highfreq_df[highfreq_df['sol_number'] == sol_number]
                #print("Loading time: ", time.time() - start_time)
                positive_entries = drop_list[drop_list[' SOL '] == sol_number]

                for index, row in positive_entries.iterrows():

                    pdrop_time = pd.to_datetime(row[' YYYY-MM-DDTHH:MM:SS.sss '])
                    pdrop_strength = row['_DROP_ ']
                    if seisdata.index[0] <= pdrop_time <= seisdata.index[-1]:

                        time_differences = abs(seisdata.index - pdrop_time)
                        closest_index = time_differences.argmin()
                        closest_entry_position = seisdata.index.get_loc(seisdata.iloc[closest_index].name)
                        start_position = max(closest_entry_position - (window_size-1)//2, 0)
                        end_position = min(closest_entry_position + (window_size)//2, len(seisdata) - 1)

                        for i in range(1, samples_per_drop+1):
                            if i % 2 == 1:
                                #offset = -pos_sample_spacing * (i//2)
                                offset = random.randint(-13, -3)
                            elif i % 2 == 0:
                                #offset = pos_sample_spacing * (i//2)
                                offset = random.randint(3, 13)

                            if start_position + offset > 0 and end_position + 1 + offset < len(seisdata): 
                                slice_around_closest = seisdata.iloc[start_position + offset:end_position + 1 + offset]
                                if 'Mid frequency data' in feature_list:
                                    midfreq_slice = midfreq_data.iloc[start_position + offset:end_position + 1 + offset]
                                if 'High frequency data' in feature_list:
                                    highfreq_slice = highfreq_data.iloc[start_position + offset:end_position + 1 + offset]


                                if np.ptp(slice_around_closest['Total']) > power_threshold:
                                    if 'Total' in feature_list:
                                        cols_to_exclude = ['sol_number']
                                    else:
                                        cols_to_exclude = ['Total','sol_number']
                                    pos_df = pos_df._append(create_sample(feature_list, slice_around_closest.copy(), cols_to_exclude=cols_to_exclude))
                                    drop_strengths.append(pdrop_strength)
                                    if 'Mid frequency data' in feature_list:
                                        pos_midfreq = pos_midfreq._append(create_sample(feature_list, midfreq_slice.copy(), cols_to_exclude=cols_to_exclude))
                                    if 'High frequency data' in feature_list:
                                        pos_highfreq = pos_highfreq._append(create_sample(feature_list, highfreq_slice.copy(), cols_to_exclude=cols_to_exclude))
                                else:
                                    if offset == 0:
                                        pDrops_below_thresh.add(pdrop_time)
                                        excluded.add(pdrop_time)

                            else:
                                pDrops_out_of_range.add(pdrop_time)

                    else:
                        if verbose > 1:
                            print(f"pdrop_time {pdrop_time} is out of range of seisdata")
                        pDrops_out_of_range.add(pdrop_time)
                        rows_to_delete.append(index)

            except Exception as e:
                if verbose > 0:
                    print(f"Error processing SOL {sol_number}: {e}")

            thresh_dict[sol_number] = excluded
    filtered_drop_list = drop_list.drop(rows_to_delete)

    if 'Mid frequency data' in feature_list:
        pos_midfreq.columns = range(pos_df.shape[1],pos_df.shape[1]+pos_midfreq.shape[1])
        pos_df = pd.concat([pos_df, pos_midfreq], axis=1)
    if 'High frequency data' in feature_list:
        pos_highfreq.columns = range(pos_df.shape[1],pos_df.shape[1]+pos_highfreq.shape[1])
        pos_df = pd.concat([pos_df, pos_highfreq], axis=1)
    if drop_magnitude == True:
        pos_df['Magnitude'] = drop_strengths

    return pos_df, pDrops_below_thresh, filtered_drop_list, pDrops_out_of_range, thresh_dict

def create_synthetic_samples(pos_df, noise_df, multiplier):

    num_synthetic_samples = int(len(pos_df) * multiplier)
    print(num_synthetic_samples, len(noise_df))
    noise_samples = noise_df.sample(n=num_synthetic_samples, replace=False)
    pos_samples = pos_df.sample(n=num_synthetic_samples, replace=True)
    noise_samples = noise_samples.set_index(pos_samples.index)
    synthetic_samples = pos_samples + noise_samples

    return synthetic_samples


def construct_neg_samples(feature_list, sol_range, test_sols, pressuredrops, power_threshold, sample_size=35, interval=35, verbose=0, noise_range=None):
    samples_df = pd.DataFrame()
    noise_df = pd.DataFrame()
    total_pdrops_excluded = set()
    total_below_thresh = 0
    file_path = './seisdata.parquet'
    midfreqdata = './seisdata_1.1Hz-3.9Hz.parquet'
    highfreqdata = 'seisdata_4.1Hz-8Hz.parquet'
    if 'Mid frequency data' in feature_list:
        midfreq_df = pd.read_parquet(midfreqdata)
    if 'High frequency data' in feature_list:
        highfreq_df = pd.read_parquet(highfreqdata)

    df = pd.read_parquet(file_path)

    for sol_number in range(sol_range[0], sol_range[1]):

        if sol_number not in test_sols:
            if verbose > 0:
                print(f"Processing SOL {sol_number}")
                start_time = time.time()
            sol_data = df[df['sol_number'] == sol_number]
            midfreq_data = None
            highfreq_data = None
            if 'Mid frequency data' in feature_list:
                midfreq_data = midfreq_df[midfreq_df['sol_number'] == sol_number]
            if 'High frequency data' in feature_list:
                highfreq_data = highfreq_df[highfreq_df['sol_number'] == sol_number]
            #print("Loading time: ", time.time() - start_time)
            sol_samples, pdrops_excluded, below_thresh, noise_samples = construct_samples(feature_list, sol_number, sol_data, pressuredrops, power_threshold, midfreq_data, highfreq_data, mode='neg', sample_size=sample_size, interval=interval, verbose=0, noise_range=noise_range)
            samples_df = pd.concat([samples_df, sol_samples], ignore_index=False)
            noise_df = pd.concat([noise_df, noise_samples], ignore_index=False)
            total_pdrops_excluded.update(pdrops_excluded)
            total_below_thresh += below_thresh
    if verbose > 0:
        print("Neg samples: Pdrops excluded, samples below threshold: ", len(total_pdrops_excluded), total_below_thresh )
    return samples_df, total_pdrops_excluded, total_below_thresh, noise_df


def predict(feature_lists, test_sols, best_clfs, model_types, feature_scalers, pressuredrops, power_threshold, sample_interval, sample_size, features, rocket_model=None, ensemble=False):

    if ensemble == False:
        feature_lists = [feature_lists]
        best_clfs = [best_clfs]
        model_types = [model_types]
        feature_scalers = [feature_scalers]

    predictions_dict = {}
    total_pdrops_excluded = set()
    total_below_thresh = 0
    last_drop_prediction = datetime(2000, 9, 21, 0, 0, 0)

    file_path = './seisdata.parquet'
    midfreqdata = './seisdata_1.1Hz-3.9Hz.parquet'
    highfreqdata = 'seisdata_4.1Hz-8Hz.parquet'
    #if 'Mid frequency data' in feature_list:
        #midfreq_df = pd.read_parquet(midfreqdata)
    #if 'High frequency data' in feature_list:
        #highfreq_df = pd.read_parquet(highfreqdata)
    df = pd.read_parquet(file_path)

    for sol_number in test_sols:
        print("Processing sol:", sol_number)
        start_time = time.time()
        sol_data = df[df['sol_number'] == sol_number]
        midfreq_data = None
        highfreq_data = None
        #if 'Mid frequency data' in feature_list:
            #midfreq_data = midfreq_df[midfreq_df['sol_number'] == sol_number]
        #if 'High frequency data' in feature_list:
            #highfreq_data = highfreq_df[highfreq_df['sol_number'] == sol_number]
        print("Loading time: ", time.time() - start_time)
        dataset, pdrops_excluded, below_thresh, _ = construct_samples(['Max vs average'], sol_number, sol_data, pressuredrops, power_threshold, midfreq_data, highfreq_data, mode='all', sample_size=sample_size, interval=sample_interval, verbose=0)
        total_pdrops_excluded.update(pdrops_excluded)
        total_below_thresh += below_thresh

        if dataset is not None and not dataset.empty:
            
            dataset.columns = dataset.columns.astype(str)
            ensemble_predictions = []
            for i, model_type in enumerate(model_types):
                feature_scaler = feature_scalers[i]
                if model_type != 'lstm':
                    CAREFUL = dataset.drop(columns='max/avg')
                else:
                    CAREFUL = dataset
                    
                feature_scaler = feature_scalers[i]
                dataset_scaled = pd.DataFrame(feature_scaler.transform(CAREFUL), columns=CAREFUL.columns, index=CAREFUL.index)
                best_clf = best_clfs[i]
                feature_list = feature_lists[i]
                if 'Time of day' in feature_list:
                    mlst_series = dataset_scaled.index.to_series().apply(lambda x: datetime_to_mlst(x))
                    dataset_scaled[['sin_time', 'cos_time']] = mlst_series.apply(cyclical_encode_time).apply(pd.Series)
                if 'Time of year' in feature_list:
                    dataset_scaled[['sin_martian_time_of_year', 'cos_martian_time_of_year']] = dataset_scaled.index.to_series().apply(cyclical_encode_martian_time_of_year).apply(pd.Series)
                if 'Time since last pDrop' not in feature_list:
                    if model_type == 'lstm':
                        predictions = (models.predict_lstm(best_clf, dataset_scaled, sample_size, features) > 0.5).astype("int32")
                    elif model_type == 'logistic rocket':
                        data, extra_features = rocket.prepare_rocket_data(dataset_scaled, predict=True)
                        rocket_features = rocket.get_rocket_features(data, sample_size, features, predict = True, rocket_model=rocket_model)
                        if len(extra_features) > 0:
                            predictions = best_clf.predict(np.hstack((rocket_features, extra_features)))
                        else:
                            predictions = best_clf.predict(rocket_features)
                    else:
                        predictions = (best_clf.predict(dataset_scaled) > 0.5).astype("int32")

                else:
                    predictions = []
                    for idx, row in dataset_scaled.iterrows():
                        if model_type == 'lstm':
                            row_reshaped = row.values.reshape((1, sample_size, features))
                            prediction = (best_clf.predict(row_reshaped) > 0.5).astype("int32")
                        else:
                            time_since_last_pDrop = int((idx - last_drop_prediction).total_seconds())
                            time_since_last_pDrop = min(time_since_last_pDrop, 86400)
                            row['time since last pDrop'] = time_since_last_pDrop
                            row_reshaped = row.values.reshape(1, -1)
                            prediction = (best_clf.predict(row_reshaped) > 0.5).astype("int32")
                        
                        if prediction == 1:
                            last_drop_prediction = idx
                        predictions.append(prediction[0])

                predictions = np.array(predictions)
                ensemble_predictions.append(predictions)

            if ensemble == True:
                #print(ensemble_predictions[0].shape, ensemble_predictions[1].shape, ensemble_predictions[2].shape)
                stacked_predictions = np.stack((ensemble_predictions[0], np.squeeze(ensemble_predictions[1]), ensemble_predictions[2]), axis=1)
                sum_predictions = np.sum(stacked_predictions, axis=1)
                predictions = np.where(sum_predictions >= 2, 1, 0)   

            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    if sol_number in predictions_dict:
                        predictions_dict[sol_number].append(dataset.index[i])
                    else: 
                        predictions_dict[sol_number] = [dataset.index[i]]

    return predictions_dict, total_pdrops_excluded, total_below_thresh



def evaluate_predictions(raw_predictions, test_sols, pressuredrops, match_threshold, sample_size):

    positive_predictions = {}
    time_offset_per_column = pd.Timedelta(seconds=2.75)
    match_time_threshold = pd.Timedelta(seconds=11.875)

    for sol_number in test_sols:
        if sol_number in raw_predictions:
            predictions = raw_predictions[sol_number]
            interval_tree = IntervalTree()
            positive_indices = set()

            for starttime in predictions:
                time_offset = time_offset_per_column * sample_size
                endtime = pd.to_datetime(starttime) + time_offset
                starttime = pd.to_datetime(starttime)
                endtime = pd.to_datetime(endtime)
                interval_tree.add(Interval(starttime.timestamp(), endtime.timestamp(), (starttime, endtime)))

            overlaps = {}
            for interval in interval_tree:
                overlapping_intervals = interval_tree.overlap(interval.begin, interval.end)
                if len(overlapping_intervals) >= match_threshold:
                    overlaps[interval.data] = overlapping_intervals

            merged_intervals = set()
            for key_interval, overlapping_intervals in overlaps.items():
                merged_interval = Interval(min(i.begin for i in overlapping_intervals), max(i.end for i in overlapping_intervals))
                #sorted_intervals = sorted(overlapping_intervals, key=lambda x: x.begin)
                #middle_index = len(sorted_intervals) // 2
                #middle_interval = sorted_intervals[middle_index]
                #merged_intervals.add(Interval(middle_interval.begin, middle_interval.end))
                merged_intervals.add(merged_interval)

            non_overlapping_intervals = set()
            for interval in sorted(merged_intervals, key=lambda x: x.end - x.begin, reverse=True):
                if not any(interval.overlaps(other) for other in non_overlapping_intervals):
                    non_overlapping_intervals.add(interval)

            for interval in non_overlapping_intervals:
                positive_indices.add((pd.to_datetime(interval.begin, unit='s'), pd.to_datetime(interval.end, unit='s')))

            positive_predictions[sol_number] = positive_indices

    TP = 0
    FP = 0
    FN = 0
    unique_matched_events = set()

    for sol_number in test_sols:

        if sol_number in positive_predictions:
            predictions = positive_predictions[sol_number]
        else:
            predictions = []
        actual_events = pressuredrops[pressuredrops[' SOL '] == sol_number][' YYYY-MM-DDTHH:MM:SS.sss ']
        actual_events_list = actual_events.to_list()
        sol_matched_events = set()

        for starttime, endtime in predictions:
            matched_events = [actual_time for actual_time in actual_events_list if starttime <= actual_time and actual_time <= endtime]
            if matched_events:
                unique_matched_events.update(matched_events)
                sol_matched_events.update(matched_events)
                TP += len(matched_events)
            else:
                FP += 1

        unmatched_events = [actual_time for actual_time in actual_events_list if all(actual_time < starttime or actual_time > endtime for starttime, endtime in predictions)]
        FN += len(unmatched_events)

        if len(unmatched_events) + len(sol_matched_events) != len(actual_events):
            print("ERROR on sol", sol_number," ! Actual events: ", len(actual_events), "Matched events + FN: ", len(unmatched_events) + len(sol_matched_events))

    unique_events_matched = len(unique_matched_events)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = unique_events_matched / (unique_events_matched + FN) if unique_events_matched + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    result = {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'unique_events_matched': unique_events_matched,
        'predictions': positive_predictions
    }

    return result


def utctoLTST(utc_date, longitude):

    # longitude must be inputted in degrees west:
    # InSight actual landing longitude: 135.623447°E.   Which is 224.376553 in degrees west
    # InSight target landing longitude: 135.97°E.       Which is 224.03 in degrees west
    # longitude is specified in a number format  eg. 224.376553

    Julian_Unix_epoch_date = datetime(1970, 1, 1, 0, 0, 0)  # JULIAN UNIX EPOCH (01/01/1970 00:00:00 UTC)
    Julian_Unix_epoch = 2440587.5

    # Julian day of reference (01 Jan 2000)
    j2000_epoch = 2451545.0
    K = 0.0009626                          # Allison's coefficient
    KNORM = 44796.0                        # Allison's normalisation factor
    Sol_ratio = 1.027491252                # Ratio Between Martian SOL and terrestrial day
    ms_day = 86400000.0                    # milliseconds in a day
    s_day = 86400.0                        # seconds in a day

    origin_date = datetime(2018, 11, 26, 5, 10, 50, 335600)  # midnight LMST of the SOL 0 USED by CHRONOS

    # Julian Date (UT)
    ts = (utc_date - Julian_Unix_epoch_date).total_seconds()  # time between input time and Julian unix epoch
    millis = ts * 1000.0
    JD_UT = Julian_Unix_epoch + (millis / ms_day)

    # UTC to TT conversion
    jday_min = 2441317.5

    jday_vals = [-2441317.5, 0., 182., 366.,
                 731., 1096., 1461., 1827.,
                 2192., 2557., 2922., 3469.,
                 3834., 4199., 4930., 5844.,
                 6575., 6940., 7487., 7852.,
                 8217., 8766., 9313., 9862.,
                 12419., 13515., 14792., 15887., 16437.]

    offset_min = 32.184

    offset_vals = [-32.184, 10., 11.0, 12.0, 13.0,
                   14.0, 15.0, 16.0, 17.0, 18.0,
                   19.0, 20.0, 21.0, 22.0, 23.0,
                   24.0, 25.0, 26.0, 27.0, 28.0,
                   29.0, 30.0, 31.0, 32.0, 33.0,
                   34.0, 35.0, 36.0, 37.0]

    if JD_UT <= jday_min + jday_vals[0]:
        TT_UTC = offset_min + offset_vals[0]
    elif JD_UT >= jday_min + jday_vals[-1]:
        TT_UTC = offset_min + offset_vals[-1]
    else:
        for i in range(len(offset_vals)):
            if (jday_min + jday_vals[i] <= JD_UT) and (jday_min + jday_vals[i + 1] > JD_UT):
                TT_UTC = offset_min + offset_vals[i]
                break

    # Julian Date (TT)
    JD_TT = JD_UT + (TT_UTC / s_day)

    # offset from J2000 epoch(TT)
    j2000_offset = JD_TT - j2000_epoch

    # Mars Mean Anomaly
    M = 19.3871 + 0.52402073 * j2000_offset

    # angle of Fiction Mean Sun
    a_FMS = (270.3871 + 0.524038496 * j2000_offset) % 360

    # perturbers
    A = [0.0071, 0.0057, 0.0039, 0.0037, 0.0021, 0.0020, 0.0018]
    tau = [2.2353, 2.7543, 1.1177, 15.7866, 2.1354, 2.4694, 32.8493]
    phi = [49.409, 168.173, 191.837, 21.736, 15.704, 95.528, 49.095]

    PBS = sum([A[i] * math.cos(((0.985626 * j2000_offset / tau[i]) + phi[i]) * math.pi / 180.) for i in range(len(A))])

    # Equation of Center
    M_rads = M * math.pi / 180
    v_M = (10.691 + 3.0e-7 * j2000_offset) * math.sin(M_rads) \
        + 0.6230 * math.sin(2 * M_rads) \
        + 0.0500 * math.sin(3 * M_rads) \
        + 0.0050 * math.sin(4 * M_rads) \
        + 0.0005 * math.sin(5 * M_rads) \
        + PBS

    # areocentric solar longitude
    L_s = a_FMS + v_M
    L_s = L_s % 360

    # Equation of Time
    ls = L_s * math.pi / 180.

    EOT = 2.861 * math.sin(2 * ls) \
        - 0.071 * math.sin(4 * ls) \
        + 0.002 * math.sin(6 * ls) - v_M

    EOT_h = EOT / 15

    # Mars Solar Date
    const = 4.5
    MSD = (((j2000_offset - const) / Sol_ratio) + KNORM - K)

    # Mean Solar Time at Mars's prime meridian,
    MST = 24 * MSD
    MST_mod = (24 * MSD) % 24

    # Local Mean Solar Time
    LMST_nomod = (MST - longitude * (24. / 360))
    LMST = ((MST_mod - longitude * (24. / 360))) % 24

    # LMST decimal to time
    hour_LMST = int(LMST)

    min_dec = 60 * (LMST - hour_LMST)
    min_LMST = int(min_dec)

    sec_dec = 60 * (min_dec - min_LMST)
    sec_LMST = int(sec_dec)

    millisec_LMST = int((sec_dec - sec_LMST) * 100000)

    # Sol - two ways
    Sol = (LMST_nomod - LMST) / 24 - 51510  # way #1

    delta_sec = (utc_date - origin_date).total_seconds()  # way #2
    martianSol = delta_sec / (s_day * Sol_ratio)

    # Local True Solar Time
    LTST = (LMST + EOT_h) % 24

    # LMST decimal to time
    hour_LTST = int(LTST)

    min_dec1 = 60 * (LTST - hour_LTST)
    min_LTST = int(min_dec1)

    sec_dec1 = 60 * (min_dec1 - min_LTST)
    sec_LTST = int(sec_dec1)

    millisec_LTST = int((sec_dec1 - sec_LTST) * 100000)

    # Output string
    output_str = f"Sol {int(martianSol)}{hour_LTST:02d}:{min_LTST:02d}:{sec_LTST:02d}.{millisec_LTST:05d} (LTST)"

    marsDate = [int(martianSol), hour_LTST, min_LTST, sec_LTST, millisec_LTST]

    return marsDate

def predict_magnitude(df_slice, classifier, window_size, feature_scaler, feature_list):

    center_index_label = df_slice['Total'].idxmax()
    center_index = df_slice.index.get_loc(center_index_label)
    half_window = window_size // 2
    if center_index - half_window < 0:
        startidx = 0
        endidx = 35 
    elif center_index + half_window > len(df_slice)-1:
        endidx = len(df_slice)
        startidx = endidx - 35
    else:
        startidx = center_index - half_window
        endidx = center_index + half_window + 1

    sample = df_slice.iloc[startidx:endidx]
    if 'Total' in feature_list:
        cols_to_exclude = ['sol_number']
    else:
        cols_to_exclude = ['sol_number', 'Total']
    sample_row = create_sample(feature_list, sample.copy(), cols_to_exclude=cols_to_exclude)
    scaled_row = pd.DataFrame(feature_scaler.transform(sample_row), columns=sample_row.columns, index=sample_row.index)
    if 'Time of Day' in feature_list:
        mlst_series = scaled_row.index.to_series().apply(lambda x: datetime_to_mlst(x))
        scaled_row[['sin_time', 'cos_time']] = mlst_series.apply(cyclical_encode_time).apply(pd.Series)
    if 'Time of Dear' in feature_list:
        scaled_row[['sin_martian_time_of_year', 'cos_martian_time_of_year']] = scaled_row.index.to_series().apply(scaled_row.cyclical_encode_martian_time_of_year).apply(pd.Series)
    #print(scaled_row)
    magnitude = classifier.predict(scaled_row)[0]
    timestamp = df_slice.index[center_index]
    LTST =  datetime_to_mlst(timestamp)
    return timestamp, LTST, magnitude


def add_magnitude_predictions(pressuredrops, directory, classifier, window_size, feature_scaler, feature_list):

    classifier = joblib.load(classifier)
    feature_scaler = joblib.load(feature_scaler)
    predictions_catalog = pd.DataFrame(columns=['Sol','Timestamp','LTST','Magnitude'])
    comparison_catalog = pd.DataFrame(columns=['Sol', 'Timestamp','LTST','Actual Strength','Predicted Strength','Predicted Timestamp'])
    with open(f'./{directory}/positive_predictions.json', 'r') as json_file:
        positive_predictions = json.load(json_file)
    print('loaded predictions')
    seisdata_path = './seisdata.parquet'
    df = pd.read_parquet(seisdata_path)
    print('loaded seisdata')
    positive_predictions = {int(key): value for key, value in positive_predictions.items()}
    for sol_number in positive_predictions.keys():
        sol_data = df[df['sol_number'] == sol_number]
        print("sol number", sol_number)
        sol_data.index = pd.to_datetime(sol_data.index)
        for interval in positive_predictions[sol_number]:
            starttime = pd.to_datetime(interval[0])
            endtime = pd.to_datetime(interval[1])
            df_slice = sol_data.loc[starttime:endtime]
            timestamp, LTST, drop_magnitude = predict_magnitude(df_slice, classifier, window_size, feature_scaler, feature_list)
            data = [sol_number, timestamp, LTST, drop_magnitude]
            predictions_catalog.loc[len(predictions_catalog)] = data
            corresponding_drop = pressuredrops[(pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] > starttime) & (pressuredrops[' YYYY-MM-DDTHH:MM:SS.sss '] < endtime)]
            if len(corresponding_drop) > 0:
                actual_magnitude = corresponding_drop.iloc[0]['_DROP_ ']
                actual_timestamp = corresponding_drop.iloc[0][' YYYY-MM-DDTHH:MM:SS.sss ']
                actual_LTST = corresponding_drop.iloc[0][' _LTST_ ']
                comparison_data = [sol_number, actual_timestamp, actual_LTST, actual_magnitude, drop_magnitude, timestamp]
                comparison_catalog.loc[len(comparison_catalog)] = comparison_data

    predictions_catalog.to_csv(f'./{directory}/predictions.csv', index=False)
    if len(comparison_catalog) > 0:
        comparison_catalog.to_csv(f'./{directory}/magnitude_comparison.csv')
        mae = mean_absolute_error(comparison_catalog['Predicted Strength'], comparison_catalog['Actual Strength'])
        print("Mean Absolute Error (MAE):", mae)
        r2 = r2_score(comparison_catalog['Predicted Strength'], comparison_catalog['Actual Strength'])
        print("R-squared (R²) Score:", r2)
        with open(f'./{directory}/log.txt', 'a') as file:
            file.write(f"Mean Absolute Error (MAE): {mae}\n")
            file.write(f"R-squared (R²) Score: {r2}\n")


def np_to_pq(data, path):
    data_df = pd.DataFrame(data)
    data_df.to_parquet(path)




    