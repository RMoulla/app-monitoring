from flask import Flask, request, jsonify
from prometheus_client import Counter, Summary, generate_latest
import pickle
import pandas as pd
import time
import logging

app = Flask(__name__)

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Charger le modèle de churn
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Modèle chargé avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle: {e}")

# Définir un compteur pour suivre le nombre total de prédictions
PREDICTIONS_COUNTER = Counter('total_predictions', 'Nombre total de prédictions')
# Résumé pour suivre le temps de traitement des requêtes
REQUEST_TIME = Summary('request_processing_seconds', 'Temps de traitement des requêtes')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()  # Mesurer le temps de la requête
def predict():
    try:
        data = request.get_json()
        logging.info(f"Données reçues: {data}")
        
        # Les features doivent correspondre aux colonnes numériques utilisées lors de l'entraînement du modèle
        features = pd.DataFrame([data])

        # Faire la prédiction à l'aide du modèle chargé
        prediction = model.predict(features)[0]
        logging.info(f"Prédiction: {prediction}")
        
        # Incrémenter le compteur de prédictions
        PREDICTIONS_COUNTER.inc()

        # Retourner la prédiction sous forme de réponse JSON
        result = {"prediction": "churn" if prediction == 1 else "no churn"}
        return jsonify(result)
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({"error": "Erreur lors du traitement de la prédiction"}), 500

# Endpoint pour exposer les métriques Prometheus
@app.route('/metrics')
def metrics():
    return generate_latest(), 200

if __name__ == '__main__':
    logging.info("Démarrage du serveur Flask")
    app.run(host='0.0.0.0', port=5000)
