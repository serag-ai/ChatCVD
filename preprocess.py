import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import random
from transformers import pipeline

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df_init = pd.read_csv(file_path)
    
    # Define cancer conditions and create HaveCancer column
    conditions = [
        (df_init['CHCSCNCR'] == 1) | (df_init['CHCOCNCR'] == 1),
        (df_init['CHCSCNCR'] == 2) | (df_init['CHCOCNCR'] == 2),
        (df_init['CHCSCNCR'] == 7) | (df_init['CHCOCNCR'] == 7),
        (df_init['CHCSCNCR'] == 9) | (df_init['CHCOCNCR'] == 9)
    ]
    choices = [1, 2, 7, 9]
    df_init['HaveCancer'] = np.select(conditions, choices, default=np.nan)
    
    return df_init

# Select relevant features
def select_features(data, categories):
    selected_features = []
    for category, features in categories.items():
        selected_features.extend(features)
    return data[selected_features]

# Undersample majority class (assuming this function is defined elsewhere)
def undersample_rus(df, target_column, majority_class, minority_class_size):
    # Implementation not provided in the original code
    pass

# Map numerical values to text descriptions
def map_features_to_text(df):
    # Define mappings for various features
    mappings = {
        '_AGEG5YR': {1: 'Age 18 to 24', 2: 'Age 25 to 29', ...},
        'SEX': {1: 'Male', 2: 'Female'},
        # Add other mappings here
    }
    
    # Apply mappings
    for column, mapping in mappings.items():
        df[f'{column}_text'] = df[column].map(mapping)
    
    return df

# Generate CVD input text
def generate_cvd_input(row):
    return f"This individual is a {row['SEX_text']} with an age between {row['_AGEG5YR_text']}. ..."

# Paraphrase text using T5 model
def paraphrase_text(text):
    paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
    paraphrases = paraphraser(f"paraphrase: {text}", num_return_sequences=1, max_length=512)
    return paraphrases[0]['generated_text']

# Main preprocessing function
def preprocess_data(file_path):
    df = load_and_preprocess_data(file_path)
    df_filtered = select_features(df, categories)
    df_filtered = df_filtered.dropna()
    
    # Preprocess specific columns
    df_filtered['_MICHD'] = df_filtered['_MICHD'].replace({2: 0})
    
    # Balance the dataset
    df_balanced = undersample_rus(df_filtered, '_MICHD', 0, 34663)
    
    # Map features to text
    df_balanced = map_features_to_text(df_balanced)
    
    # Generate CVD input and risk
    df_balanced['CVD_Input'] = df_balanced.apply(generate_cvd_input, axis=1)
    df_balanced['CVD_Risk'] = df_balanced['_MICHD'].map({0: 'Low Risk', 1: 'High Risk'})
    
    # Paraphrase CVD input
    df_balanced['CVD_Input_Paraphrased'] = df_balanced['CVD_Input'].apply(paraphrase_text)
    
    return df_balanced

# Run preprocessing
if __name__ == "__main__":
    preprocessed_data = preprocess_data('/path/to/your/data.csv')
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)
