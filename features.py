import numpy as np
def add_date_features(df, col = 'pickup_datetime'):
    df['year'] = df[col].dt.year
    df['month'] = df[col].dt.month
    df['day_of_month'] = df[col].dt.day
    df['day_of_week'] = df[col].dt.weekday
    df['hour'] = df[col].dt.hour
    df['is_weekend'] = df[col].dt.weekday.isin([5,6]).astype('uint8')
    df['is_night'] = ((df[col].dt.hour < 6) | (df[col].dt.hour > 22)).astype('uint8')
    df['rush_hour'] = df[col].dt.hour.isin([7,8,9,16,17,18]).astype('uint8')
    df['week'] = df[col].dt.isocalendar().week.astype('int16')
    df['quarter'] = df[col].dt.quarter
    return df

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Haversine distance in km
    lat/lon inputs can be scalars, pandas Series, or NumPy arrays
    """
    R = 6371.0  # Earth radius in km
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def add_trip_distance(df):
    df['trip_distance_km'] = haversine_np(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )
    return df

def add_is_manhattan(df):
    df['pickup_in_manhattan'] = (
        (df['pickup_longitude'] > -74.03) &
        (df['pickup_longitude'] < -73.93) &
        (df['pickup_latitude'] > 40.70) &
        (df['pickup_latitude'] < 40.85)
    ).astype('uint8')
    
    df['dropoff_in_manhattan'] = (
        (df['dropoff_longitude'] > -74.03) &
        (df['dropoff_longitude'] < -73.93) &
        (df['dropoff_latitude'] > 40.70) &
        (df['dropoff_latitude'] < 40.85)
    ).astype('uint8')
    
    return df

jfk_lonlat = (-73.7781, 40.6413)
lga_lonlat = (-73.8740, 40.7769)
ewr_lonlat = (-74.1745, 40.6895)
met_lonlat = (-73.9632, 40.7794)
wtc_lonlat = (-74.0099, 40.7126)

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(
        df['dropoff_latitude'], df['dropoff_longitude'],
        lat, lon
    )

def add_landmarks(df):
    landmarks = [
        ('jfk', jfk_lonlat),
        ('lga', lga_lonlat),
        ('ewr', ewr_lonlat),
        ('met', met_lonlat),
        ('wtc', wtc_lonlat)
    ]
    for name, lonlat in landmarks:
        add_landmark_dropoff_distance(df, name, lonlat)
    return df

def cross_manhattan(df):
    df['trip_crosses_manhattan'] = (df['pickup_in_manhattan'] != df['dropoff_in_manhattan']).astype('uint8')
    return df

def add_cross_features(df):
    df["rush_hour_x_distance"] = df["rush_hour"] * df["trip_distance_km"]
    df["cross_manhattan_x_distance"] = df["trip_crosses_manhattan"] * df["trip_distance_km"]
    df["weekend_x_distance"] = df["is_weekend"] * df["trip_distance_km"]
    return df

def add_min_landmark_distance(df):
    landmark_cols = [
        'jfk_drop_distance', 
        'lga_drop_distance', 
        'ewr_drop_distance', 
        'met_drop_distance', 
        'wtc_drop_distance'
    ]
    
    df["min_landmark_distance"] = df[landmark_cols].min(axis = 1)
    
    return df

def add_outlier_flags(df):
    """Flag very short and very long trips"""
    df["is_short_trip"] = (df["trip_distance_km"] < 1).astype('uint8')
    df["is_long_trip"] = (df["trip_distance_km"] > 30).astype('uint8')

def apply_features(df, datetime_col = "pickup_datetime"):
    add_date_features(df, col = datetime_col)
    add_trip_distance(df)
    add_is_manhattan(df)
    add_landmarks(df)
    cross_manhattan(df)
    add_cross_features(df)
    add_min_landmark_distance(df)
    add_outlier_flags(df)
    return df