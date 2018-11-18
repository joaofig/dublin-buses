import numpy as np
import pandas as pd


def pandas_load_day(day):
    header = ['timestamp', 'line_id', 'direction', 'jrny_patt_id', 'time_frame', 'journey_id', 'operator',
              'congestion', 'lon', 'lat', 'delay', 'block_id', 'vehicle_id', 'stop_id', 'at_stop']
    types = {'timestamp': np.int64,
             'journey_id': np.int32,
             'congestion': np.int8,
             'lon': np.float64,
             'lat': np.float64,
             'delay': np.int8,
             'vehicle_id': np.int32,
             'at_stop': np.int8}
    file_name = 'data/siri.201301{0:02d}.csv'.format(day)
    df = pd.read_csv(file_name, header=None, names=header, dtype=types, parse_dates=['time_frame'],
                     infer_datetime_format=True)
    null_replacements = {'line_id': 0, 'stop_id': 0}
    df = df.fillna(value=null_replacements)
    df['line_id'] = df['line_id'].astype(np.int32)
    df['stop_id'] = df['stop_id'].astype(np.int32)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    return df


def run():
    for i in range(1, 32):
        print(i)
        day = pandas_load_day(i)
        stops = day.loc[day['at_stop'] == 1, ['lat', 'lon', 'stop_id']]

        file_name = 'data/stops{0:02d}.csv'.format(i)
        stops.to_csv(file_name, index=False)


if __name__ == '__main__':
    run()
