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


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    Taken from here: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas#29546836
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = 6378137.0 * c
    return meters


def get_move_ability_array(source: np.ndarray, index: int, window=4) -> np.ndarray:
    source_size = source.shape[0]
    n = 2 * window + 1
    if 0 <= index < window:
        # The index lies on the left window
        result = np.full(n, source[0])
        result[window - index:] = source[:window + index + 1]
    elif source_size - window <= index < source_size:
        # The index lies in the right window
        result = np.full(n, source[-1])
        a = source_size - index - 1
        result[:window + a + 1] = source[index - window:index + a + 1]
    else:
        result = source[index - window:index + window + 1]
    return result


def trajectory_curve_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    distance = 0.0
    for i in range(1, lats.shape[0]):
        distance += haversine_np(lons[i-1], lats[i-1], lons[i], lats[i])
    return distance


def trajectory_direct_distance(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    return haversine_np(longitudes[0], latitudes[0], longitudes[-1], latitudes[-1])


def get_move_ability(df: pd.DataFrame, columns=['lat', 'lon']) -> np.ndarray:
    locations = np.transpose(df[columns].values)
    move_ability = []
    for i in range(locations.shape[1]):
        lats = get_move_ability_array(locations[0, :], i)
        lons = get_move_ability_array(locations[1, :], i)

        curve_dist = trajectory_curve_distance(lats, lons)
        direct_dist = trajectory_direct_distance(lats, lons)

        if curve_dist > 0.0:
            move_ability.append(direct_dist / curve_dist)
        else:
            move_ability.append(0)

    return np.array(move_ability)


def run():
    # a = np.arange(100)
    # for i in range(100):
    #     print(i, get_move_ability_array(a, i))
    # for i in range(1, 32):
    #     print(i)
    #     day = pandas_load_day(i)
    #     stops = day.loc[day['at_stop'] == 1, ['lat', 'lon', 'stop_id']]
    #
    #     file_name = 'data/stops{0:02d}.csv'.format(i)
    #     stops.to_csv(file_name, index=False)

    df = pandas_load_day(1)
    df['move_ability'] = 0.0

    vehicles = df['vehicle_id'].unique()

    for v in vehicles:
        vehicle = df[df['vehicle_id'] == v]
        move_ability = get_move_ability(vehicle)
        df.loc[df['vehicle_id'] == v, 'move_ability'] = move_ability


if __name__ == '__main__':
    run()
