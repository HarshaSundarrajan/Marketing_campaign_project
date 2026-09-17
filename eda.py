import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("cleaned_data/cleaned_marketing_campaign.csv")

print("Dataset Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Brand Revenue:\n", df.groupby("Brand")["Revenue"].sum().sort_values(ascending=False))
print("Campaign Revenue:\n", df.groupby("Campaign_Type")["Revenue"].mean().sort_values(ascending=False))
print("Channel ROI:\n", df.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=False).head(10))

plt.figure(figsize=(8, 5))
df.groupby("Brand")["Revenue"].sum().plot(kind="bar")
plt.title("Total Revenue by Brand")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("plots/revenue_by_brand.png")
plt.close()

plt.figure(figsize=(10, 5))
df.groupby("Campaign_Type")["ROI"].mean().plot(kind="bar")
plt.title("Average ROI by Campaign Type")
plt.ylabel("Average ROI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/roi_by_campaign_type.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df["Revenue"], bins=30)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/revenue_distribution.png")
plt.close()

numeric_cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 7))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("EDA completed. Charts saved in plots folder.")
