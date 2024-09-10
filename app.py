from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle de churn
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Données reçues : {data}")
        
        # Convertir les données en DataFrame
        features = pd.DataFrame([data])
        
        # Prédire le churn à l'aide du modèle
        prediction = model.predict(features)[0]
        
        # Retourner la prédiction sous forme de réponse JSON
        result = {"prediction": "churn" if prediction == 1 else "no churn"}
        return jsonify(result)
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": "Erreur lors de la prédiction"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
