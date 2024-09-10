from flask import Flask, request, jsonify
import pickle
import pandas as pd
from prometheus_client import Counter, generate_latest, Summary

app = Flask(__name__)

# Charger le modèle de churn
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Créer des métriques Prometheus
PREDICTIONS_COUNTER = Counter('total_predictions', 'Nombre total de prédictions')
REQUEST_TIME = Summary('request_processing_seconds', 'Temps de traitement des requêtes')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()  # Mesurer le temps de traitement de la requête
def predict():
    try:
        data = request.get_json()  # Récupérer les données envoyées dans la requête
        print(f"Données reçues : {data}")
        
        # Convertir les données en DataFrame pour le modèle
        features = pd.DataFrame([data])
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        # Incrémenter le compteur de prédictions
        PREDICTIONS_COUNTER.inc()
        
        # Retourner la prédiction sous forme de réponse JSON
        result = {"prediction": "churn" if prediction == 1 else "no churn"}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route pour exposer les métriques Prometheus
@app.route('/metrics')
def metrics():
    return generate_latest(), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
