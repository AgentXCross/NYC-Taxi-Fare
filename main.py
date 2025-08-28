from data import load_data
from model import train_model

def main():
    #Load data 
    X_train, y_train, categorical_columns = load_data("taxi-nyc/train.csv")

    #Train model and save the best model using joblib
    model = train_model(X_train, y_train, categorical_columns)

if __name__ == "__main__":
    main()