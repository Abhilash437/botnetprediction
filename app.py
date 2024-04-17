import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses, Sequential

scaler = MinMaxScaler()

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(115, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(37, activation="relu"),
            layers.Dense(28, activation="relu")
        ])
        self.decoder = Sequential([
            layers.Dense(37, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(115, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def predict(x, ae, threshold=0.0068555507, window_size=82):
    x = scaler.transform(x)
    predictions = losses.mse(x, ae.predict(x)) > threshold
    # Majority voting over `window_size` predictions
    return np.array([np.mean(predictions[i-window_size:i]) > 0.5
                     for i in range(window_size, len(predictions)+1)])

def print_stats(data, outcome):
    st.subheader(f"Shape of data: {data.shape}")
    st.subheader(f"Detected anomalies: {np.mean(outcome)*100}%")

def load_nbaiot(filename):
    return np.loadtxt((filename), delimiter=",", skiprows=1)

st.title('Botnet prediction app')
st.write("""-- This app predicts botnet in the dataset --""")

st.sidebar.header('Please Input the file')

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_df = load_nbaiot(uploaded_file)
    
    st.write(input_df)

    # Fit the scaler on the input data
    scaler.fit(input_df)

    # Recreate the model architecture
    new_model = Autoencoder()
    # Call the model to build its variables
    new_model(np.zeros((1, input_df.shape[1])))

    # Load the weights (assuming 'my_autoencoder_weights.h5' is the saved weights file)
    new_model.load_weights('my_autoencoder_weights.h5')

    if st.button("Click here to Predict type of Botnet"):
        result = predict(input_df, new_model)
        print_stats(input_df, result)
        # You can modify this section to interpret the 'result' for anomaly detection

else:
    st.info('Please upload a CSV file for prediction.')
