CREATE DATABASE IF NOT EXISTS marketing_campaign_db;
USE marketing_campaign_db;

DROP TABLE IF EXISTS marketing_campaigns;

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
    Revenue DECIMAL(15,2),
    Acquisition_Cost DECIMAL(15,2),
    ROI DECIMAL(10,2),
    Language VARCHAR(50),
    Engagement_Score DECIMAL(10,2),
    Customer_Segment VARCHAR(100),
    Date DATE,
    Brand VARCHAR(50),
    Year INT,
    Month INT,
    Profit_Flag INT
);
