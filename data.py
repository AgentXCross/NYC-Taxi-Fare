import pandas as pd
from sklearn.model_selection import train_test_split
import random
from features import apply_features

dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}
sample_fraction = 0.1
random.seed(42)
def skip_row(row_index):
    if row_index == 0:
        return False
    return random.random() > sample_fraction
random.seed(42)

def load_data(train_path, test_path):
    df = pd.read_csv(
        train_path, 
        parse_dates = ['pickup_datetime'], 
        date_format = '%Y-%m-%d %H:%M:%S %Z',
        dtype = dtypes,
        skiprows = skip_row
        )
    #Apply Feature Engineering
    df = apply_features(df)

    #Split training data and get useful columns
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

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

    #Remove values that don't make sense
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

    #Extract only needed columns from training and validation dataframe
    X_train = train_df[categorical_cols + numeric_cols]
    y_train = train_df[target_cols]
    X_valid = valid_df[categorical_cols + numeric_cols]
    y_valid = valid_df[target_cols]

    #Apply same transformations to testing set
    test_df = pd.read_csv(test_path, parse_dates = ['pickup_datetime'], date_format = '%Y-%m-%d %H:%M:%S %Z', dtype = dtypes)
    test_df = apply_features(test_df)
    X_test = test_df[categorical_cols + numeric_cols]

    return X_train, y_train, X_valid, y_valid, X_test, categorical_cols