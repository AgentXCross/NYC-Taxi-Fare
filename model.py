import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def train_model(X_train, y_train, categorical_cols):
    params = {
        "objective": "regression",
        "boosting_type": 'gbdt',
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 45,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "verbose": -1,
        'min_gain_to_split': 0.01,
        'max_depth': 10
    }

    model = LGBMRegressor(
        **params,
        n_estimators = 420, 
        n_jobs = -1, 
        random_state = 77
    )

    model.fit(
        X_train, y_train,
        categorical_feature = categorical_cols
    )

    joblib.dump(model, "lgbm_model.pkl")
    joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
    return model

def load_model(path = "lgbm_model.pkl"):
    return joblib.load(path)