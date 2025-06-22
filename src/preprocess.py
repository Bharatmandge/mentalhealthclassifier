import pandas as pd
import os

def load_and_clean_data():
    # File path
    file_path = r"data/raw/data_to_be_cleansed.csv"
    
    # Read the dataset
    df = pd.read_csv(file_path)

    # Initial Checks
    print("Shape:", df.shape)
    print("Info:\n", df.info())
    print("Missing values:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())

    # Basic Cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"] = df["text"].str.lower()

    return df

def save_cleaned_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned.csv", index=False)
    print("✅ Cleaned data saved at: data/processed/cleaned.csv")

if __name__ == "__main__":
    df = load_and_clean_data()
    save_cleaned_data(df)

