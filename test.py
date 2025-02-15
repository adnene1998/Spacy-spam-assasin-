import streamlit as st
import spacy
from joblib import load
import numpy as np

# Charger le modèle SpaCy et le modèle ML
nlp = spacy.load("en_core_web_sm")
loaded_clf = load("random_forest_model.joblib")  # Charger une seule fois

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# Titre de l'application
st.title("Spam Detection with Streamlit")

# Champ de saisie
user_input = st.text_input("Enter some text:")

if user_input:
    processed_text = preprocess(user_input)
    st.write("Processed text:", processed_text)

    # Vérification que le texte n'est pas vide après prétraitement
    if processed_text.strip():
        doc = nlp(processed_text)
        vector = doc.vector.reshape(1, -1)  # Transformation correcte

        # Prédiction
        result = loaded_clf.predict(vector)
        response = ["Ham", "Spam"]
        st.write("Prediction:", response[int(result[0])])
    else:
        st.write("Not enough meaningful words for prediction.")
else:
    st.write("Please enter some text.")
