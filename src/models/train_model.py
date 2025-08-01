import pandas as pd
import joblib
import os

def main():
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # Charger le modèle complet sauvegardé par gridsearch
    best_model = joblib.load('models/best_model.pkl')

    # Réentraîner le modèle sur toutes les données (optionnel, ou juste réutiliser)
    best_model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    # Sauvegarder le modèle sous le nom attendu dans DVC
    joblib.dump(best_model, 'models/model.pkl')

if __name__ == '__main__':
    main()
