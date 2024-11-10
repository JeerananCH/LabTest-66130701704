import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders from the pickle file
@st.cache
def load_model():
    with open('model_penguin_66130701704.pkl', 'rb') as file:
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
    return model, species_encoder, island_encoder, sex_encoder

# Function to predict the species
def predict_penguin_species(model, species_encoder, island_encoder, sex_encoder, island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g):
    # Create DataFrame from user input
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]  # Ensure sex is in lowercase
    })
    
    # Apply encoding to categorical columns
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])  # Transform 'male' or 'female'
    
    # Make prediction
    prediction = model.predict(x_new)
    predicted_species = species_encoder.inverse_transform(prediction)
    
    return predicted_species[0]

# Streamlit UI
def app():
    # Load model and encoders
    model, species_encoder, island_encoder, sex_encoder = load_model()

    # Title and description
    st.title("Penguin Species Prediction")
    st.write("This app predicts the species of a penguin based on physical characteristics.")

    # Sidebar for user input
    st.sidebar.header("Input Features")

    # Input fields
    island = st.sidebar.selectbox("Island", ["Torgersen", "Biscoe", "Dream"])
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])  # User inputs 'Male' or 'Female'
    culmen_length_mm = st.sidebar.slider("Culmen Length (mm)", 30.0, 70.0, 45.0)
    culmen_depth_mm = st.sidebar.slider("Culmen Depth (mm)", 10.0, 25.0, 15.0)
    flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", 150.0, 250.0, 200.0)
    body_mass_g = st.sidebar.slider("Body Mass (g)", 2500, 6500, 4000)

    # Button to trigger prediction
    if st.sidebar.button("Predict"):
        # Get prediction
        x_new = pd.DataFrame({
            'island': [island],
            'culmen_length_mm': [culmen_length_mm],
            'culmen_depth_mm': [culmen_depth_mm],
            'flipper_length_mm': [flipper_length_mm],
            'body_mass_g': [body_mass_g],
            'sex': [sex.upper()]  # Ensure sex is in lowercase
        })
        
        # Apply encoding to categorical columns
        x_new['island'] = island_encoder.transform(x_new['island'])
        x_new['sex'] = sex_encoder.transform(x_new['sex'])  # Transform 'male' or 'female'
        st.write("### Input Data")
        st.dataframe(x_new)  # Display the DataFrame
        # Make prediction
        prediction = model.predict(x_new)
        predicted_species = species_encoder.inverse_transform(prediction)
        
        
        
        # Display result
        st.write(f"The predicted penguin species is **{ predicted_species[0]}**.")

# Run the app
if __name__ == "__main__":
    app()
