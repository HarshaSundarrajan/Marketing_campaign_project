import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

os.makedirs("models", exist_ok=True)

df = pd.read_csv("cleaned_data/cleaned_marketing_campaign.csv")
# Use a representative sample for faster training on normal laptops.
# Remove this line if you want to train on the full dataset.
df = df.sample(n=min(30000, len(df)), random_state=42)

# Multi-label encoding for Channel_Used
channel_lists = df["Channel_Used"].fillna("Unknown").apply(lambda x: [i.strip() for i in str(x).split(",")])
mlb = MultiLabelBinarizer()
channel_encoded = pd.DataFrame(
    mlb.fit_transform(channel_lists),
    columns=["Channel_" + c for c in mlb.classes_],
    index=df.index
)
df = pd.concat([df, channel_encoded], axis=1)
joblib.dump(mlb, "models/channel_encoder.pkl")

numeric_features = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost", "Engagement_Score", "Year", "Month"]
categorical_features = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
channel_features = list(channel_encoded.columns)

features = numeric_features + categorical_features + channel_features

X = df[features]
y_reg = df["Revenue"]
y_cls = df["Profit_Flag"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("channel", "passthrough", channel_features),
    ]
)

# ---------------- Regression Models ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=20, max_depth=12, random_state=42, n_jobs=-1)
}

best_reg_model = None
best_reg_score = -999
reg_results = []

for name, model in reg_models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    reg_results.append([name, mae, mse, rmse, r2])
    if r2 > best_reg_score:
        best_reg_score = r2
        best_reg_model = pipe

joblib.dump(best_reg_model, "models/revenue_model.pkl")

# ---------------- Classification Models ----------------
# Important: ROI, Calculated_ROI, Revenue are not used as features to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

cls_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=20, max_depth=12, random_state=42, n_jobs=-1)
}

best_cls_model = None
best_cls_score = -999
cls_results = []

for name, model in cls_models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cls_results.append([name, acc, precision, recall, f1])
    if f1 > best_cls_score:
        best_cls_score = f1
        best_cls_model = pipe

joblib.dump(best_cls_model, "models/profit_model.pkl")
joblib.dump(features, "models/features.pkl")

pd.DataFrame(reg_results, columns=["Model", "MAE", "MSE", "RMSE", "R2"]).to_csv("models/regression_results.csv", index=False)
pd.DataFrame(cls_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1_Score"]).to_csv("models/classification_results.csv", index=False)

print("Model training completed")
print("Best regression R2:", best_reg_score)
print("Best classification F1:", best_cls_score)
print("Models saved inside models folder")
