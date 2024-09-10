from flask import Flask, request, jsonify
from prometheus_client import Counter, Summary, generate_latest
import pickle
import pandas as pd
import time

app = Flask(__name__)

# Charger le modèle de churn
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Définir un compteur pour suivre le nombre total de prédictions
PREDICTIONS_COUNTER = Counter('total_predictions', 'Nombre total de prédictions')
# Résumé pour suivre le temps de traitement des requêtes
REQUEST_TIME = Summary('request_processing_seconds', 'Temps de traitement des requêtes')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()  # Mesurer le temps de la requête
def predict():
    data = request.get_json()
    # Les features doivent correspondre aux colonnes numériques utilisées lors de l'entraînement du modèle
    features = pd.DataFrame([data])
    
    # Faire la prédiction à l'aide du modèle chargé
    prediction = model.predict(features)[0]
    
    # Incrémenter le compteur de prédictions
    PREDICTIONS_COUNTER.inc()

    # Retourner la prédiction sous forme de réponse JSON
    result = {"prediction": "churn" if prediction == 1 else "no churn"}
    return jsonify(result)

# Endpoint pour exposer les métriques Prometheus
@app.route('/metrics')
def metrics():
    return generate_latest(), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
