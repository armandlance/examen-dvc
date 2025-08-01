import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    model = joblib.load('models/model.pkl')  # <-- changer ici

    predictions = model.predict(X_test)

    os.makedirs('metrics', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {'mse': mse, 'r2': r2}
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f)

    pd.DataFrame({'prediction': predictions}).to_csv('data/predictions.csv', index=False)

if __name__ == '__main__':
    main()
