import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Charger le dataset churn
df = pd.read_csv('customer_churn.csv')

# Sélectionner uniquement les colonnes numériques
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# On suppose que la colonne 'Churn' est la variable cible (elle doit être convertie en 0 et 1 si elle ne l'est pas)
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    y = df['Churn']
else:
    raise ValueError("La colonne 'Churn' n'existe pas dans le dataset.")

# Supprimer la colonne cible des variables explicatives
X = numerical_df.drop(columns=['Churn'])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner un modèle RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions et évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Sauvegarder le modèle dans un fichier .pkl
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
