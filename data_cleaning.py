import os
import pandas as pd
import numpy as np

DATA_FILES = {
    "Tira": "data/tira_campaign.csv",
    "Purplle": "data/purplle_campaign.csv",
    "Nykaa": "data/nykaa_campaign.csv",
}

NUMERIC_COLS = [
    "Duration", "Impressions", "Clicks", "Leads", "Conversions",
    "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score"
]
CATEGORICAL_COLS = [
    "Campaign_ID", "Campaign_Type", "Target_Audience", "Channel_Used",
    "Language", "Customer_Segment"
]

def load_and_merge():
    frames = []
    for brand, path in DATA_FILES.items():
        df = pd.read_csv(path)
        df["Brand"] = brand
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Date"] = df["Date"].fillna(df["Date"].mode()[0])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # ROI validation: if ROI missing/impossible, recalculate using revenue and cost.
    # Formula: ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost
    df["Calculated_ROI"] = np.where(
        df["Acquisition_Cost"] != 0,
        (df["Revenue"] - df["Acquisition_Cost"]) / df["Acquisition_Cost"],
        0
    )
    df["ROI"] = df["ROI"].replace([np.inf, -np.inf], np.nan).fillna(df["Calculated_ROI"])

    # Profit/Loss target
    df["Profit_Flag"] = (df["ROI"] > 0).astype(int)

    return df

if __name__ == "__main__":
    os.makedirs("cleaned_data", exist_ok=True)
    df = load_and_merge()
    cleaned = clean_data(df)
    cleaned.to_csv("cleaned_data/cleaned_marketing_campaign.csv", index=False)
    print("Cleaning completed")
    print("Shape:", cleaned.shape)
    print("Saved: cleaned_data/cleaned_marketing_campaign.csv")
