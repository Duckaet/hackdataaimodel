from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load all necessary files
model = joblib.load('disease_predictor_model_xgb.pkl')
scaler = joblib.load('scaler.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_country = joblib.load('label_encoder_country.pkl')
label_encoder_disease = joblib.load('label_encoder_disease.pkl')
pca = joblib.load('pca.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON
        data = request.json
        
        # Extract fields from the JSON input
        age = data.get('Age')
        gender = data.get('Gender')
        country = data.get('Country')
        symptoms = f"{data.get('Symptom 1', '')} {data.get('Symptom 2', '')} {data.get('Symptom 3', '')}"

        # Validate input
        if not all([age, gender, country, symptoms.strip()]):
            return jsonify({'error': 'All fields (Age, Gender, Country, Symptoms) are required'}), 400

        # Preprocess input data
        gender_encoded = label_encoder_gender.transform([gender])[0]
        country_encoded = label_encoder_country.transform([country])[0]
        symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
        numeric_features = scaler.transform([[age, gender_encoded, country_encoded]])
        final_features = np.hstack((numeric_features, symptoms_vectorized))
        final_features_reduced = pca.transform(final_features)

        # Predict disease
        prediction = model.predict(final_features_reduced)
        predicted_disease = label_encoder_disease.inverse_transform(prediction)[0]

        # Return the prediction as a JSON response
        return jsonify({
            'Predicted Disease': predicted_disease
        })

    except Exception as e:
        # Handle exceptions gracefully
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
