import pandas as pd
import numpy as np
import random
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. Train the Machine Learning Model with XGBoost and TF-IDF Vectorization
def train_model():
    # Load the dataset
    data = pd.read_csv('disease.csv')

    # Handle missing values
    data.fillna('', inplace=True)

    # Encode categorical features (Gender, Country)
    label_encoder_gender = LabelEncoder()
    label_encoder_country = LabelEncoder()
    label_encoder_disease = LabelEncoder()  # Label encoder for Disease (target variable)
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    data['Country'] = label_encoder_country.fit_transform(data['Country'])
    
    # Encode the target variable (Disease)
    data['Disease'] = label_encoder_disease.fit_transform(data['Disease'])

    # Combine symptoms into a single feature
    data['Symptoms'] = data['Symptom 1'] + ' ' + data['Symptom 2'] + ' ' + data['Symptom 3']

    # Prepare features and labels
    X = data[['Age', 'Gender', 'Country', 'Symptoms']]
    y = data['Disease']

    # TF-IDF vectorization of symptoms
    vectorizer = TfidfVectorizer(max_features=5000)
    X_symptoms = vectorizer.fit_transform(X['Symptoms']).toarray()

    # Normalize age and other numeric features
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X[['Age', 'Gender', 'Country']])

    # Combine the processed numeric and text features
    X_final = np.hstack((X_numeric, X_symptoms))

    # Reduce dimensionality with PCA for faster predictions
    pca = PCA(n_components=min(50, X_final.shape[1]))  # Adjust components based on available features
    X_reduced = pca.fit_transform(X_final)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # Train an XGBoost Classifier with hyperparameter tuning
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Multi-class classification
        eval_metric='mlogloss',
        random_state=42
    )

    # Hyperparameter tuning with GridSearchCV
    params = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]
    }
    grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model after tuning
    best_model = grid_search.best_estimator_

    # Save the trained model and preprocessing objects
    joblib.dump(best_model, 'disease_predictor_model_xgb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(label_encoder_gender, 'label_encoder_gender.pkl')
    joblib.dump(label_encoder_country, 'label_encoder_country.pkl')
    joblib.dump(label_encoder_disease, 'label_encoder_disease.pkl')  # Save the disease encoder
    joblib.dump(pca, 'pca.pkl')

    print("Model training complete and saved.")

# 2. Chatbot Logic: Load the Trained Model and Interact with the User
def chatbot():
    # Load the trained model and preprocessing objects
    model = joblib.load('disease_predictor_model_xgb.pkl')
    scaler = joblib.load('scaler.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder_gender = joblib.load('label_encoder_gender.pkl')
    label_encoder_country = joblib.load('label_encoder_country.pkl')
    label_encoder_disease = joblib.load('label_encoder_disease.pkl')  # Load the disease encoder
    pca = joblib.load('pca.pkl')

    print("Welcome to the Healthcare Chatbot!")
    print("Please provide your details to start the consultation.")
    
    # Collecting User Input
    try:
        age = int(input("Enter your age: "))
        gender = input("Enter your gender (Male/Female): ")
        country = input("Enter your country: ")
    except ValueError:
        print("Invalid input. Please enter valid age and gender.")
        return

    # Validate gender and country inputs
    if gender not in ['Male', 'Female']:
        print("Invalid gender. Please enter 'Male' or 'Female'.")
        return
    if not country:
        print("Country cannot be empty.")
        return
    
    # Symptoms
    print("Please describe your symptoms. You can describe up to 3 symptoms.")
    symptom_1 = input("Symptom 1: ")
    symptom_2 = input("Symptom 2: ")
    symptom_3 = input("Symptom 3: ")
    
    # Preprocessing user input
    user_data = {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'Symptom 1': symptom_1,
        'Symptom 2': symptom_2,
        'Symptom 3': symptom_3
    }
    
    # Process symptoms
    user_symptoms = symptom_1 + " " + symptom_2 + " " + symptom_3
    user_input = pd.DataFrame([user_data])
    
    # Preprocess data for model prediction
    try:
        user_input['Gender'] = label_encoder_gender.transform(user_input['Gender'])
        user_input['Country'] = label_encoder_country.transform(user_input['Country'])
    except KeyError as e:
        print(f"Error encoding categorical data: {e}")
        return
    
    # Vectorize symptoms and scale other features
    user_symptoms_vectorized = vectorizer.transform([user_symptoms]).toarray()
    user_numeric_features = scaler.transform(user_input[['Age', 'Gender', 'Country']])
    user_final_features = np.hstack((user_numeric_features, user_symptoms_vectorized))
    
    # Reduce dimensionality
    user_final_features_reduced = pca.transform(user_final_features)

    # Predict disease
    predicted_disease_numeric = model.predict(user_final_features_reduced)
    predicted_disease = label_encoder_disease.inverse_transform(predicted_disease_numeric)  # Decode numeric to original label
    print(f"Based on your symptoms, the chatbot predicts the disease as: {predicted_disease[0]}")
    
    # Give possible treatment suggestions (optional)
    treatment_suggestions = ["Consult a specialist", "Rest and stay hydrated", "Take prescribed medication"]
    print(f"Suggested treatment: {random.choice(treatment_suggestions)}")

# 3. Main Program Execution
if __name__ == "__main__":
    choice = input("Would you like to (1) Train the model or (2) Start the chatbot? Enter 1 or 2: ")

    if choice == '1':
        train_model()  # Train and save the model
    elif choice == '2':
        chatbot()  # Start the chatbot and make predictions
    else:
        print("Invalid choice. Exiting.")
