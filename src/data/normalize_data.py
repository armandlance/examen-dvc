import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    # Chargement des données
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')

    # Sélection uniquement des colonnes numériques
    X_train_numeric = X_train.select_dtypes(include=['number'])
    X_test_numeric = X_test.select_dtypes(include=['number'])

    scaler = StandardScaler()

    # Normalisation
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Conversion en DataFrame pour conserver les noms de colonnes
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

    # Sauvegarde
    X_train_scaled_df.to_csv('data/processed/X_train_scaled.csv', index=False)
    X_test_scaled_df.to_csv('data/processed/X_test_scaled.csv', index=False)

if __name__ == '__main__':
    main()
