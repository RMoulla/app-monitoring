from flask import Flask, request, jsonify
import pickle
import pandas as pd
from prometheus_client import Counter, generate_latest, Summary
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.debug("Démarrage de l'application Flask")

# Charger le modèle de churn
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.debug("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")

# Créer des métriques Prometheus
PREDICTIONS_COUNTER = Counter('total_predictions', 'Nombre total de prédictions')
REQUEST_TIME = Summary('request_processing_seconds', 'Temps de traitement des requêtes')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time() # Mesurer le temps de traitement de la requête
def predict():
    try:
        data = request.get_json() # Récupérer les données envoyées dans la requête
        logger.debug(f"Données reçues : {data}")
        # Convertir les données en DataFrame pour le modèle
        features = pd.DataFrame([data])
        # Faire la prédiction
        prediction = model.predict(features)[0]
        # Incrémenter le compteur de prédictions
        PREDICTIONS_COUNTER.inc()
        # Retourner la prédiction sous forme de réponse JSON
        result = {"prediction": "churn" if prediction == 1 else "no churn"}
        logger.debug(f"Prédiction effectuée : {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route pour exposer les métriques Prometheus
@app.route('/metrics')
def metrics():
    logger.debug("Requête reçue sur /metrics")
    return generate_latest(), 200

if __name__ == '__main__':
    logger.debug("Démarrage du serveur Flask")
    app.run(host='0.0.0.0', port=8000)