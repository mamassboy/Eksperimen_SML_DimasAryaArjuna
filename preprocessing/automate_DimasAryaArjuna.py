import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "TelcoChurn_preprocessing"
)

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_telco(df: pd.DataFrame):
    df = df.copy()

    # Cleaning TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Target
    df["label"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop kolom yang tidak dipakai
    df = df.drop(columns=["Churn", "customerID"])

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in df.columns 
                            if col not in numeric_features + ["label"]]

    X = df.drop(columns=["label"])
    y = df["label"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    final_df = pd.concat(
        [X_processed_df.reset_index(drop=True),
         y.reset_index(drop=True)],
        axis=1
    )

    return final_df, preprocessor

def save_outputs(df_processed: pd.DataFrame, preprocessor):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_csv = os.path.join(OUTPUT_DIR, "telco_churn_preprocessed.csv")
    df_processed.to_csv(output_csv, index=False)

    preprocessor_path = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
    dump(preprocessor, preprocessor_path)

    print("Preprocessed data saved to:", output_csv)
    print("Preprocessor saved to:", preprocessor_path)

def main():
    print("Loading raw data from:", RAW_DATA_PATH)
    df_raw = load_raw_data(RAW_DATA_PATH)
    print("Raw shape:", df_raw.shape)

    df_processed, preprocessor = preprocess_telco(df_raw)
    print("Processed shape:", df_processed.shape)

    save_outputs(df_processed, preprocessor)

if __name__ == "__main__":
    main()
