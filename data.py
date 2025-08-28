import pandas as pd
import random
from features import apply_features

#Data type for each column
dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}

#Only extract 50 percent of the data
sample_fraction = 0.5
random.seed(42)
def skip_row(row_index):
    """Function for only pulling part of the dataset as the data set is very large"""
    if row_index == 0:
        return False
    return random.random() > sample_fraction
random.seed(42)

def load_data(train_path):
    """Loads training data. Only used when model hyperparameters are optimized."""
    df = pd.read_csv(
        train_path, 
        parse_dates = ['pickup_datetime'], 
        date_format = '%Y-%m-%d %H:%M:%S %Z',
        dtype = dtypes,
        skiprows = skip_row
        )
    #Apply Feature Engineering
    df = apply_features(df)

    #Column Spliting
    numeric_cols = [
    'year', 'passenger_count',
    'is_weekend', 'is_night', 'rush_hour',
    'trip_distance_km', 'pickup_in_manhattan', 'dropoff_in_manhattan',
    'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
    'met_drop_distance', 'wtc_drop_distance', 'trip_crosses_manhattan',
    'is_short_trip', 'is_long_trip', 'min_landmark_distance', 'rush_hour_x_distance',
    'cross_manhattan_x_distance', 'weekend_x_distance'
    ]

    categorical_cols = ['month', 'quarter', 'day_of_month', 'day_of_week', 'hour', 'week']

    target_cols = ['fare_amount']

    #Remove values that don't make sense from the dataframe
    df = df.loc[
        df['fare_amount'].between(0, 200) &
        df['passenger_count'].between(1, 6) &
        df['pickup_latitude'].between(40.5, 41.0) &
        df['dropoff_latitude'].between(40.5, 41.0) &
        df['pickup_longitude'].between(-74.3, -73.6) &
        df['dropoff_longitude'].between(-74.3, -73.6)
    ]
    df = df.loc[df['fare_amount'] > 0]
    df = df.loc[~(
        (df['pickup_latitude'] == df['dropoff_latitude']) &
        (df['pickup_longitude'] == df['dropoff_longitude'])
    )]

    #Extract only needed columns from training dataframe
    X_train = df[categorical_cols + numeric_cols]
    y_train = df[target_cols]

    return X_train, y_train, categorical_cols