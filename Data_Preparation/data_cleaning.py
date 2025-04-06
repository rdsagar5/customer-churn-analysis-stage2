# data_cleaning.py

import pandas as pd

def load_data(file_path):
   
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print("File not found")
        return None

def clean_data(df):
    
    print(f"Initial shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop rows with missing values (or modify to fill missing values)
    df = df.dropna()
    
    print(f"Cleaned shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_csv = "/Users/sagarbk/Documents/WIL/customer_churn_raw.csv"
    output_csv = "/Users/sagarbk/Documents/WIL/customer_churn_cleaned.csv"

    df_raw = load_data(input_csv)

    if df_raw is not None:
        df_clean = clean_data(df_raw)
        save_cleaned_data(df_clean, output_csv)