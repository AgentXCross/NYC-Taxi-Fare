import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def train_model(X_train, y_train, X_valid, y_valid):
    categorical_cols = ['month', 'quarter', 'day_of_month', 'day_of_week', 'hour', 'week']

    model = LGBMRegressor(
        boosting_type = 'gbdt',
        objective = 'regression',
        metric = 'rmse',
        n_estimators = 500,
        learning_rate = 0.1,
        num_leaves = 31,
        max_depth = -1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        n_jobs = -1,
        min_split_gain = 0.5,
        min_child_weight = 1,
        min_child_samples = 10
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        eval_metric = 'rmse',
        categorical_feature = categorical_cols,
    )

    print("Validation RMSE:", rmse(y_valid, model.predict(X_valid)))
    joblib.dump(model, "best_model.pkl")
    joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
    return model

def load_model(path = "best_model.pkl"):
    return joblib.load(path)