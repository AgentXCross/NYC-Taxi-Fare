import pandas as pd
from sklearn.model_selection import train_test_split
import random
from features import add_date_features, add_trip_distance, add_is_manhattan, add_landmarks, cross_manhattan

dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'dropoff_latitude': 'float32',
    'passenger_count': 'uint8'
}
sample_fraction = 0.01
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
    df = add_date_features(df, 'pickup_datetime')
    df = add_trip_distance(df)
    df = add_is_manhattan(df)
    df = add_landmarks(df)
    df = cross_manhattan(df)

    #Split training data and get useful columns
    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

    numeric_cols = [
        'year', 'passenger_count',
        'is_weekend', 'is_night', 'rush_hour',
        'trip_distance_km', 'pickup_in_manhattan', 'dropoff_in_manhattan',
        'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
        'met_drop_distance', 'wtc_drop_distance', 'trip_crosses_manhattan'
    ]
    categorical_cols = ['month', 'quarter', 'day_of_month', 'day_of_week', 'hour', 'week']
    target_col = 'fare_amount'

    #Remove values that don't make any sense
    df = df.dropna()
    df = df[df['passenger_count'] < 6]
    df = df[
        (df['pickup_latitude'].between(35.0, 45.0)) &
        (df['dropoff_latitude'].between(35.0, 45.0)) &
        (df['pickup_longitude'].between(-80.0, -65.0)) &
        (df['dropoff_longitude'].between(-80.0, -65.0))
    ]
    df = df[df['fare_amount'].between(0.5, 200)]
    df = df[df['trip_distance_km'].between(0.05, 100)]

    #Extract only needed columns from training and validation dataframe
    X_train = train_df[categorical_cols + numeric_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[categorical_cols + numeric_cols]
    y_valid = valid_df[target_col]

    #Apply same transformations to testing set
    test_df = pd.read_csv(test_path, parse_dates = ['pickup_datetime'], date_format = '%Y-%m-%d %H:%M:%S %Z', dtype = dtypes)
    test_df = add_date_features(test_df, 'pickup_datetime')
    test_df = add_trip_distance(test_df)
    test_df = add_is_manhattan(test_df)
    test_df = add_landmarks(test_df)
    test_df = cross_manhattan(test_df)
    X_test = test_df[categorical_cols + numeric_cols]

    #Ensure these columns will be interpreted as categorical
    for c in ['month','quarter','day_of_month','day_of_week','hour','week']:
        X_train[c] = X_train[c].astype('int16')
        X_valid[c] = X_valid[c].astype('int16')
        X_test[c] = X_test[c].astype('int16')

    return X_train, y_train, X_valid, y_valid, X_test