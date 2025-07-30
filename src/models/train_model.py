import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main():
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    best_params = joblib.load('models/best_params.pkl')

    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')

if __name__ == '__main__':
    main()
