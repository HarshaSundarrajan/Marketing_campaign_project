import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #e0f2fe);
    color: #111827;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #312e81, #1e1b4b, #111827);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-baseweb="select"] * {
    color: #111827 !important;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    color: #1e1b4b;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #4338ca;
    margin-bottom: 30px;
    font-weight: 600;
}
.card {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(79,70,229,0.30);
    text-align: center;
}
.card h3 {
    color: #e0e7ff;
    font-size: 16px;
}
.card h1 {
    color: white;
    font-size: 30px;
    font-weight: 900;
}
h1, h2, h3 {
    color: #1e1b4b;
}
.footer {
    text-align: center;
    color: #4338ca;
    font-weight: 600;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    files = []

    for folder in ["data", "cleaned_data"]:
        if os.path.exists(folder):
            files.extend(glob.glob(f"{folder}/*.csv"))

    if len(files) == 0:
        st.error("No CSV files found. Put your datasets inside data/ or cleaned_data/ folder.")
        st.stop()

    dfs = []

    for file in files:
        temp = pd.read_csv(file)
        filename = os.path.basename(file).lower()
        temp.columns = temp.columns.str.strip()

        if "Brand" not in temp.columns:
            if "nykaa" in filename:
                temp["Brand"] = "Nykaa"
            elif "purplle" in filename:
                temp["Brand"] = "Purplle"
            elif "tira" in filename:
                temp["Brand"] = "Tira"
            elif "Campaign_ID" in temp.columns and temp["Campaign_ID"].notna().any():
                first_id = str(temp["Campaign_ID"].dropna().iloc[0]).upper()
                if first_id.startswith("NY"):
                    temp["Brand"] = "Nykaa"
                elif first_id.startswith("PU"):
                    temp["Brand"] = "Purplle"
                elif first_id.startswith("TI"):
                    temp["Brand"] = "Tira"
                else:
                    temp["Brand"] = "Unknown"
            else:
                temp["Brand"] = "Unknown"

        dfs.append(temp)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates()

    numeric_cols = [
        "Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [
        "Campaign_Type", "Target_Audience", "Channel_Used",
        "Language", "Customer_Segment", "Brand"
    ]

    for col in categorical_cols:
        if col in df.columns:
            fill_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year.fillna(0).astype(int)
        df["Month"] = df["Date"].dt.month.fillna(0).astype(int)

    if "Revenue" in df.columns and "Acquisition_Cost" in df.columns:
        df["Calculated_ROI"] = np.where(
            df["Acquisition_Cost"] != 0,
            (df["Revenue"] - df["Acquisition_Cost"]) / df["Acquisition_Cost"],
            0
        )

        if "ROI" in df.columns:
            df["ROI_Difference"] = abs(df["ROI"] - df["Calculated_ROI"])
            df["ROI_Valid"] = np.where(df["ROI_Difference"] <= 0.10, "Valid", "Mismatch")
        else:
            df["ROI"] = df["Calculated_ROI"]
            df["ROI_Valid"] = "Calculated"

    if "ROI" in df.columns:
        df["Profit_Flag"] = df["ROI"].apply(lambda x: 1 if x > 0 else 0)

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def apply_channel_encoding(df):
    df = df.copy()

    if "Channel_Used" not in df.columns:
        df["Channel_Used"] = "Unknown"

    existing_channel_cols = [
        col for col in df.columns
        if col.startswith("Channel_") and col != "Channel_Used"
    ]

    df = df.drop(columns=existing_channel_cols, errors="ignore")

    channel_clean = (
        df["Channel_Used"]
        .astype(str)
        .str.replace("/", ",", regex=False)
        .str.replace("|", ",", regex=False)
        .str.replace(";", ",", regex=False)
        .str.replace("+", ",", regex=False)
        .str.replace("&", ",", regex=False)
    )

    channel_encoded = channel_clean.str.get_dummies(sep=",")

    channel_encoded.columns = [
        "Channel_" + col.strip().replace(" ", "_")
        for col in channel_encoded.columns
    ]

    channel_encoded = channel_encoded.loc[:, ~channel_encoded.columns.duplicated()]
    df = pd.concat([df, channel_encoded], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


df_raw = load_data()
df = clean_data(df_raw)
df = apply_channel_encoding(df)
df = df.loc[:, ~df.columns.duplicated()]


@st.cache_resource
def train_models(data):
    model_df = data.copy()
    model_df = model_df.loc[:, ~model_df.columns.duplicated()]

    required_cols = ["Revenue", "Profit_Flag"]

    for col in required_cols:
        if col not in model_df.columns:
            st.error(f"Required column missing: {col}")
            st.stop()

    drop_cols = ["Campaign_ID", "Date", "Channel_Used", "ROI_Valid"]
    model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns], errors="ignore")
    model_df = model_df.loc[:, ~model_df.columns.duplicated()]

    regression_features = [
        col for col in model_df.columns
        if col not in ["Revenue", "Profit_Flag"]
    ]

    classification_features = [
        col for col in model_df.columns
        if col not in ["Profit_Flag", "ROI", "Calculated_ROI", "ROI_Difference", "Revenue"]
    ]

    X_reg = model_df[regression_features]
    y_reg = model_df["Revenue"]

    X_cls = model_df[classification_features]
    y_cls = model_df["Profit_Flag"]

    reg_cat = X_reg.select_dtypes(include=["object"]).columns.tolist()
    reg_num = X_reg.select_dtypes(include=["int64", "float64"]).columns.tolist()

    cls_cat = X_cls.select_dtypes(include=["object"]).columns.tolist()
    cls_num = X_cls.select_dtypes(include=["int64", "float64"]).columns.tolist()

    reg_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), reg_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), reg_cat)
    ])

    cls_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), cls_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cls_cat)
    ])

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=30, random_state=42)
    }

    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=30, random_state=42)
    }

    reg_results = []
    trained_reg_models = {}

    for name, model in regression_models.items():
        pipe = Pipeline([
            ("preprocessor", reg_preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train_r, y_train_r)
        pred = pipe.predict(X_test_r)

        reg_results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test_r, pred),
            "MSE": mean_squared_error(y_test_r, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_r, pred)),
            "R2 Score": r2_score(y_test_r, pred)
        })

        trained_reg_models[name] = pipe

    cls_results = []
    trained_cls_models = {}

    for name, model in classification_models.items():
        pipe = Pipeline([
            ("preprocessor", cls_preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train_c, y_train_c)
        pred = pipe.predict(X_test_c)

        cls_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test_c, pred),
            "Precision": precision_score(y_test_c, pred, zero_division=0),
            "Recall": recall_score(y_test_c, pred, zero_division=0),
            "F1 Score": f1_score(y_test_c, pred, zero_division=0)
        })

        trained_cls_models[name] = pipe

    return (
        pd.DataFrame(reg_results),
        pd.DataFrame(cls_results),
        trained_reg_models,
        trained_cls_models,
        regression_features,
        classification_features
    )


reg_results, cls_results, reg_models, cls_models, reg_features, cls_features = train_models(df)


st.sidebar.title("📈 Marketing Dashboard")
st.sidebar.markdown("### Multi-Brand Campaign Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 EDA",
        "🏆 Brand Analysis",
        "💡 Business Insights",
        "🤖 Revenue Prediction",
        "💰 Profit/Loss Prediction",
        "📈 Model Evaluation",
        "🗄️ SQL Queries",
        "📋 Dataset",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

filtered_df = df.copy()

if "Brand" in df.columns:
    brands = sorted(df["Brand"].dropna().unique())
    selected_brands = st.sidebar.multiselect("Brand", brands, default=brands)
    filtered_df = filtered_df[filtered_df["Brand"].isin(selected_brands)]

if "Campaign_Type" in df.columns:
    campaigns = sorted(df["Campaign_Type"].dropna().unique())
    selected_campaigns = st.sidebar.multiselect("Campaign Type", campaigns, default=campaigns)
    filtered_df = filtered_df[filtered_df["Campaign_Type"].isin(selected_campaigns)]

if "Customer_Segment" in df.columns:
    segments = sorted(df["Customer_Segment"].dropna().unique())
    selected_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)
    filtered_df = filtered_df[filtered_df["Customer_Segment"].isin(selected_segments)]


st.markdown(
    '<div class="main-title">📈 Multi-Brand Marketing Campaign Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Nykaa | Purplle | Tira | Python | SQL | Machine Learning | Streamlit</div>',
    unsafe_allow_html=True
)


def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <h1>{value}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


if page == "🏠 Dashboard":
    total_revenue = filtered_df["Revenue"].sum()
    avg_roi = filtered_df["ROI"].mean()
    total_campaigns = len(filtered_df)
    profit_campaigns = filtered_df["Profit_Flag"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    with c2:
        kpi_card("📈 Average ROI", f"{avg_roi:.2f}")
    with c3:
        kpi_card("📢 Total Campaigns", f"{total_campaigns:,}")
    with c4:
        kpi_card("✅ Profit Campaigns", f"{profit_campaigns:,}")

    st.markdown("## 📊 Campaign Overview")

    col1, col2 = st.columns(2)

    with col1:
        brand_rev = filtered_df.groupby("Brand")["Revenue"].sum().reset_index()
        fig = px.bar(brand_rev, x="Brand", y="Revenue", color="Brand", text_auto=True,
                     title="Brand-wise Revenue")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        camp_rev = filtered_df.groupby("Campaign_Type")["Revenue"].sum().reset_index()
        fig = px.pie(camp_rev, names="Campaign_Type", values="Revenue",
                     title="Revenue by Campaign Type", hole=0.45)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "Date" in filtered_df.columns:
            trend = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
            fig = px.line(trend, x="Date", y="Revenue", title="Revenue Trend")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(filtered_df, x="ROI", nbins=40, title="ROI Distribution")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🏆 Top & Low Performing Campaigns")

    col5, col6 = st.columns(2)

    with col5:
        top_campaigns = filtered_df.sort_values("Revenue", ascending=False).head(10)
        fig = px.bar(top_campaigns, x="Campaign_ID", y="Revenue", color="Brand",
                     title="Top 10 Campaigns by Revenue")
        fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        low_campaigns = filtered_df.sort_values("Revenue", ascending=True).head(10)
        fig = px.bar(low_campaigns, x="Campaign_ID", y="Revenue", color="Brand",
                     title="Lowest 10 Campaigns by Revenue")
        fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📡 Channel-wise Effectiveness")

    channel_cols = [
        col for col in filtered_df.columns
        if col.startswith("Channel_") and col != "Channel_Used"
    ]

    if len(channel_cols) > 0:
        channel_data = []

        for col in channel_cols:
            temp = filtered_df[filtered_df[col] == 1]
            channel_data.append({
                "Channel": col.replace("Channel_", ""),
                "Revenue": temp["Revenue"].sum(),
                "Clicks": temp["Clicks"].sum(),
                "Conversions": temp["Conversions"].sum(),
                "Average ROI": temp["ROI"].mean()
            })

        channel_df = pd.DataFrame(channel_data)

        fig = px.bar(channel_df, x="Channel", y="Revenue", color="Channel",
                     text_auto=True, title="Channel-wise Revenue Effectiveness")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(channel_df, use_container_width=True)


elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(filtered_df, x="Clicks", y="Revenue", color="Brand",
                         title="Clicks vs Revenue")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(filtered_df, x="Acquisition_Cost", y="Revenue", color="Brand",
                         title="Acquisition Cost vs Revenue")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        seg = filtered_df.groupby("Customer_Segment")["Revenue"].sum().reset_index()
        fig = px.bar(seg, x="Customer_Segment", y="Revenue", color="Customer_Segment",
                     title="Customer Segment Revenue")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        lang = filtered_df.groupby("Language")["Revenue"].sum().reset_index()
        fig = px.bar(lang, x="Language", y="Revenue", color="Language",
                     title="Language-wise Revenue")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🔥 Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


elif page == "🏆 Brand Analysis":
    st.markdown("## 🏆 Brand Performance Analysis")

    brand_summary = filtered_df.groupby("Brand").agg({
        "Revenue": "sum",
        "ROI": "mean",
        "Clicks": "sum",
        "Conversions": "sum",
        "Acquisition_Cost": "sum"
    }).reset_index()

    st.dataframe(brand_summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(brand_summary, x="Brand", y="Revenue", color="Brand",
                     text_auto=True, title="Total Revenue by Brand")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(brand_summary, x="Brand", y="ROI", color="Brand",
                     text_auto=True, title="Average ROI by Brand")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


elif page == "💡 Business Insights":
    st.markdown("## 💡 Business Insights & Recommendations")

    best_brand = filtered_df.groupby("Brand")["Revenue"].sum().idxmax()
    best_brand_revenue = filtered_df.groupby("Brand")["Revenue"].sum().max()

    best_roi_brand = filtered_df.groupby("Brand")["ROI"].mean().idxmax()
    best_roi_value = filtered_df.groupby("Brand")["ROI"].mean().max()

    best_campaign_type = filtered_df.groupby("Campaign_Type")["Revenue"].sum().idxmax()
    best_campaign_revenue = filtered_df.groupby("Campaign_Type")["Revenue"].sum().max()

    profit_count = filtered_df["Profit_Flag"].sum()
    total_campaigns = len(filtered_df)
    profit_percentage = (profit_count / total_campaigns) * 100 if total_campaigns > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("🏆 Best Brand", best_brand)
    with c2:
        kpi_card("📈 Best ROI Brand", best_roi_brand)
    with c3:
        kpi_card("🎯 Best Campaign", best_campaign_type)
    with c4:
        kpi_card("✅ Profit %", f"{profit_percentage:.2f}%")

    st.markdown("## 📌 Key Insights")

    st.success(f"{best_brand} generated the highest revenue of ₹{best_brand_revenue:,.2f}.")
    st.info(f"{best_roi_brand} achieved the highest average ROI of {best_roi_value:.2f}.")
    st.warning(f"{best_campaign_type} is the most effective campaign type based on revenue.")

    st.markdown("## 📡 Best Marketing Channel")

    channel_cols = [
        col for col in filtered_df.columns
        if col.startswith("Channel_") and col != "Channel_Used"
    ]

    if len(channel_cols) > 0:
        channel_data = []

        for col in channel_cols:
            temp = filtered_df[filtered_df[col] == 1]
            channel_data.append({
                "Channel": col.replace("Channel_", ""),
                "Revenue": temp["Revenue"].sum(),
                "Clicks": temp["Clicks"].sum(),
                "Conversions": temp["Conversions"].sum(),
                "Average ROI": temp["ROI"].mean()
            })

        channel_df = pd.DataFrame(channel_data)
        best_channel = channel_df.sort_values("Revenue", ascending=False).iloc[0]["Channel"]

        st.success(f"{best_channel} is the most effective channel based on total revenue.")

        fig = px.bar(channel_df, x="Channel", y="Revenue", color="Channel",
                     text_auto=True, title="Channel-wise Revenue Comparison")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## ✅ Data-driven Recommendations")

    st.markdown(f"""
    - Focus more marketing budget on **{best_brand}**, because it produces the highest revenue.
    - Improve campaign strategies similar to **{best_campaign_type}**, because it performs well in revenue generation.
    - Give priority to **{best_roi_brand}**, because it gives better return on investment.
    - Increase investment in the best-performing marketing channels.
    - Reduce or redesign campaigns with low revenue and low ROI.
    - Use the Revenue Prediction and Profit/Loss Prediction models before launching new campaigns.
    - Track acquisition cost carefully because high cost can reduce profitability.
    """)


elif page == "🤖 Revenue Prediction":
    st.markdown("## 🤖 Revenue Prediction")

    best_reg_name = reg_results.sort_values("RMSE").iloc[0]["Model"]
    best_reg_model = reg_models[best_reg_name]

    st.success(f"Best Regression Model Selected: {best_reg_name}")

    sample_input = df[reg_features].iloc[[0]].copy()

    for col in sample_input.columns:
        if pd.api.types.is_numeric_dtype(sample_input[col]):
            sample_input[col] = st.number_input(
                col, value=float(sample_input[col].iloc[0]), key=f"reg_{col}"
            )
        else:
            options = sorted(df[col].dropna().unique()) if col in df.columns else [sample_input[col].iloc[0]]
            sample_input[col] = st.selectbox(col, options, index=0, key=f"reg_{col}")

    if st.button("Predict Revenue"):
        prediction = best_reg_model.predict(sample_input)[0]
        st.success(f"Predicted Revenue: ₹{prediction:,.2f}")


elif page == "💰 Profit/Loss Prediction":
    st.markdown("## 💰 Profit / Loss Prediction")

    st.warning("ROI and Revenue are excluded from classification input features to avoid data leakage.")

    best_cls_name = cls_results.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    best_cls_model = cls_models[best_cls_name]

    st.success(f"Best Classification Model Selected: {best_cls_name}")

    sample_input = df[cls_features].iloc[[0]].copy()

    for col in sample_input.columns:
        if pd.api.types.is_numeric_dtype(sample_input[col]):
            sample_input[col] = st.number_input(
                col, value=float(sample_input[col].iloc[0]), key=f"cls_{col}"
            )
        else:
            options = sorted(df[col].dropna().unique()) if col in df.columns else [sample_input[col].iloc[0]]
            sample_input[col] = st.selectbox(col, options, index=0, key=f"cls_{col}")

    if st.button("Predict Profit / Loss"):
        prediction = best_cls_model.predict(sample_input)[0]

        if prediction == 1:
            st.success("Prediction Result: PROFIT Campaign ✅")
        else:
            st.error("Prediction Result: LOSS Campaign ❌")


elif page == "📈 Model Evaluation":
    st.markdown("## 📈 Model Evaluation Report")

    st.markdown("### Regression Model Evaluation")
    st.dataframe(reg_results, use_container_width=True)

    fig = px.bar(reg_results, x="Model", y="RMSE", color="Model",
                 title="Regression Model RMSE Comparison")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification Model Evaluation")
    st.dataframe(cls_results, use_container_width=True)

    fig = px.bar(cls_results, x="Model", y="Accuracy", color="Model",
                 title="Classification Model Accuracy Comparison")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


elif page == "🗄️ SQL Queries":
    st.markdown("## 🗄️ SQL & MySQL Queries")

    sql_queries = """
CREATE DATABASE marketing_campaign_db;
USE marketing_campaign_db;

CREATE TABLE marketing_campaigns (
    Campaign_ID VARCHAR(50),
    Campaign_Type VARCHAR(100),
    Target_Audience VARCHAR(100),
    Duration INT,
    Channel_Used VARCHAR(255),
    Impressions BIGINT,
    Clicks BIGINT,
    Leads BIGINT,
    Conversions BIGINT,
    Revenue DOUBLE,
    Acquisition_Cost DOUBLE,
    ROI DOUBLE,
    Language VARCHAR(100),
    Engagement_Score DOUBLE,
    Customer_Segment VARCHAR(100),
    Date DATE,
    Brand VARCHAR(100),
    Profit_Flag INT
);

SELECT COUNT(*) AS total_campaigns FROM marketing_campaigns;

SELECT Brand, SUM(Revenue) AS total_revenue
FROM marketing_campaigns
GROUP BY Brand;

SELECT Brand, AVG(ROI) AS average_roi
FROM marketing_campaigns
GROUP BY Brand;

SELECT Campaign_Type, SUM(Revenue) AS total_revenue
FROM marketing_campaigns
GROUP BY Campaign_Type;

SELECT Customer_Segment, SUM(Revenue) AS revenue
FROM marketing_campaigns
GROUP BY Customer_Segment;

SELECT Language, SUM(Revenue) AS revenue
FROM marketing_campaigns
GROUP BY Language;

SELECT Brand, SUM(Clicks) AS total_clicks
FROM marketing_campaigns
GROUP BY Brand;

SELECT Brand, SUM(Conversions) AS total_conversions
FROM marketing_campaigns
GROUP BY Brand;

SELECT Campaign_ID, Revenue
FROM marketing_campaigns
ORDER BY Revenue DESC
LIMIT 10;

SELECT Campaign_ID, ROI
FROM marketing_campaigns
ORDER BY ROI DESC
LIMIT 10;

SELECT Brand, COUNT(*) AS profit_campaigns
FROM marketing_campaigns
WHERE Profit_Flag = 1
GROUP BY Brand;

SELECT Brand, COUNT(*) AS loss_campaigns
FROM marketing_campaigns
WHERE Profit_Flag = 0
GROUP BY Brand;

SELECT AVG(Acquisition_Cost) AS avg_cost
FROM marketing_campaigns;

SELECT Brand, AVG(Acquisition_Cost) AS avg_cost
FROM marketing_campaigns
GROUP BY Brand;

SELECT Campaign_Type, AVG(Engagement_Score) AS avg_engagement
FROM marketing_campaigns
GROUP BY Campaign_Type;

SELECT Brand, SUM(Revenue) - SUM(Acquisition_Cost) AS profit
FROM marketing_campaigns
GROUP BY Brand;

SELECT Date, SUM(Revenue) AS daily_revenue
FROM marketing_campaigns
GROUP BY Date
ORDER BY Date;

SELECT Brand, Campaign_Type, SUM(Revenue) AS revenue
FROM marketing_campaigns
GROUP BY Brand, Campaign_Type;

SELECT Brand, Customer_Segment, SUM(Revenue) AS revenue
FROM marketing_campaigns
GROUP BY Brand, Customer_Segment;

SELECT Campaign_ID, Clicks, Revenue
FROM marketing_campaigns
ORDER BY Clicks DESC
LIMIT 10;

SELECT Campaign_ID, Conversions, Revenue
FROM marketing_campaigns
ORDER BY Conversions DESC
LIMIT 10;

SELECT Brand, MAX(Revenue) AS max_revenue
FROM marketing_campaigns
GROUP BY Brand;

SELECT Brand, MIN(Revenue) AS min_revenue
FROM marketing_campaigns
GROUP BY Brand;

SELECT Brand, AVG(Revenue) AS avg_revenue
FROM marketing_campaigns
GROUP BY Brand;

SELECT Campaign_Type, COUNT(*) AS total
FROM marketing_campaigns
GROUP BY Campaign_Type;

SELECT Brand, ROUND(SUM(Revenue) / SUM(Acquisition_Cost), 2) AS revenue_cost_ratio
FROM marketing_campaigns
GROUP BY Brand;

SELECT Campaign_ID, Revenue, ROI,
RANK() OVER(ORDER BY Revenue DESC) AS revenue_rank
FROM marketing_campaigns;

SELECT Brand, Date, SUM(Revenue) AS revenue
FROM marketing_campaigns
GROUP BY Brand, Date;

SELECT Brand,
CASE 
    WHEN AVG(ROI) > 1 THEN 'High Performing'
    WHEN AVG(ROI) BETWEEN 0 AND 1 THEN 'Medium Performing'
    ELSE 'Low Performing'
END AS performance_category
FROM marketing_campaigns
GROUP BY Brand;

CREATE VIEW brand_performance AS
SELECT Brand, SUM(Revenue) AS total_revenue, AVG(ROI) AS avg_roi
FROM marketing_campaigns
GROUP BY Brand;
"""
    st.code(sql_queries, language="sql")


elif page == "📋 Dataset":
    st.markdown("## 📋 Dataset Preview")

    st.success(f"Rows: {filtered_df.shape[0]} | Columns: {filtered_df.shape[1]}")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.markdown("## Missing Values")
    missing = filtered_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

    st.markdown("## Multi-label Encoded Channel Columns")
    channel_cols = [col for col in df.columns if col.startswith("Channel_") and col != "Channel_Used"]
    st.write(channel_cols)


elif page == "ℹ️ About":
    st.markdown("""
    ## ℹ️ About This Project

    **Project Title:** Multi-Brand Marketing Campaign Performance Analysis and Prediction Using Python, SQL, and Machine Learning

    This application performs:

    - CSV to DataFrame conversion
    - Data cleaning
    - Missing value handling
    - Duplicate removal
    - ROI validation
    - Profit/Loss feature creation
    - Multi-label encoding for Channel_Used
    - Exploratory Data Analysis
    - Brand-wise marketing analysis
    - Top and low campaign performance analysis
    - Channel-wise effectiveness analysis
    - Business insights and recommendations
    - Revenue regression modeling
    - Profit/Loss classification modeling
    - Model evaluation
    - SQL query generation
    - Streamlit dashboard deployment

    ### Developed by

    **Harsha Sundarrajan**
    """)

st.markdown("---")
st.markdown(
    '<div class="footer">🚀 Marketing Campaign Performance Prediction | Python | SQL | Machine Learning | Streamlit</div>',
    unsafe_allow_html=True
)