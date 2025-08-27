from data import load_data
from model import train_model

def main():
    #Load data (train/valid/test)
    X_train, y_train, X_valid, y_valid, X_test = load_data(
        "taxi-nyc/train.csv",
        "taxi-nyc/test.csv"
    )

    #Train model and save best_model.pkl
    model = train_model(X_train, y_train, X_valid, y_valid)

if __name__ == "__main__":
    main()