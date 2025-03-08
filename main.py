import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
        page_title="insulin_switch_app",
)

# Charger les modèles
modele_folder = Path(__file__).parent / "model"

model = joblib.load(modele_folder / "best_logistic_model_pipeline.joblib")

# Dictionnaire pour traduire les variables en anglais
variable_labels = {
    'sexe': 'sexe_men',
    'sedentaire': 'sedentary',
    'diabete type 2': 'family history of type 2 diabetes',
    'spupd': 'discovered by spupd',
    'decompensation': 'discovered by ketoacidosis decompensation',
    'sulfamide_seule': 'initial treatment sulfamide alone'

}

# Liste des colonnes dans l'ordre attendu par le modèle
expected_features = ['sexe', 'sedentaire', 'spupd',  'decompensation',  'sulfamide_seule']


# Liste des variables binaires
binary_variables =['sexe', 'sedentaire', 'spupd',  'decompensation',  'sulfamide_seule']

# Interface utilisateur
st.title("Application of the Research Work: 'Predicting Transition to Insulin Therapy in Type 2 Diabetes Using Machine Learning'")

# Sous-titre avec le nom du professeur
st.subheader("Directed by Professor Ines Khochtali")


binary_inputs = {}
for var in binary_variables:
    binary_inputs[var] = st.selectbox(f"{variable_labels[var]} (Yes/No)", ['No', 'Yes'])


# Prétraitement des entrées utilisateur
input_data = {}
for var in expected_features:
    if var in binary_inputs:
        input_data[var] = 1 if binary_inputs[var] == 'Yes' else 0
 

# Transformer en DataFrame
input_df = pd.DataFrame([input_data])

# Prédiction
if st.button("Predict"):
    try:
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = "switch to insulin" if probability > 0.42 else "no switch to insulin"

        st.write(f"Prediction: {prediction}")
        st.write(f"Probability of switch to insulin: {probability:.2f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")


# Séparation
st.markdown('---')

# Section "Authors"
st.subheader('Authors')
st.write("Application created by Resident Mohamed Kenani under the guidance of Professor Ines Khochtali and Dr. Ekram Hajji")

# Séparation avant la section Contact
st.markdown('---')
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Section "Contact"
st.subheader('Contact')
container6 = st.container()

contact_form = """
<form action="https://formsubmit.co/kenanimohamed19@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" required placeholder="Your name" required>
     <input type="email" name="email" required placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here" required></textarea>
     <button type="submit">Send</button>
</form>"""
left_column, right_column = container6.columns(2)
left_column.markdown(contact_form, unsafe_allow_html=True)
right_column.empty()
