# Multi-Brand Marketing Campaign Performance Analysis and Prediction

## Objective
Analyze Nykaa, Purplle and Tira campaign performance and build ML models to predict Revenue and Profit/Loss.

## Run Steps
```bash
pip install -r requirements.txt
python data_cleaning.py
python eda.py
python model_training.py
streamlit run app.py
```

## Deliverables
- Cleaned CSV
- EDA charts
- SQL scripts and 30 queries
- Revenue prediction model
- Profit/Loss prediction model
- Streamlit dashboard

## Important Note
ROI is not used as an input feature for Profit/Loss classification to avoid data leakage.
