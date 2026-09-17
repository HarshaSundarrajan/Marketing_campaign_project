USE marketing_campaign_db;

-- 1 Total records
SELECT COUNT(*) AS total_records FROM marketing_campaigns;
-- 2 Total revenue
SELECT SUM(Revenue) AS total_revenue FROM marketing_campaigns;
-- 3 Brand wise revenue
SELECT Brand, SUM(Revenue) AS total_revenue FROM marketing_campaigns GROUP BY Brand ORDER BY total_revenue DESC;
-- 4 Brand wise average ROI
SELECT Brand, AVG(ROI) AS avg_roi FROM marketing_campaigns GROUP BY Brand ORDER BY avg_roi DESC;
-- 5 Campaign type wise revenue
SELECT Campaign_Type, SUM(Revenue) AS total_revenue FROM marketing_campaigns GROUP BY Campaign_Type ORDER BY total_revenue DESC;
-- 6 Top 10 campaigns by revenue
SELECT Campaign_ID, Brand, Revenue FROM marketing_campaigns ORDER BY Revenue DESC LIMIT 10;
-- 7 Bottom 10 campaigns by ROI
SELECT Campaign_ID, Brand, ROI FROM marketing_campaigns ORDER BY ROI ASC LIMIT 10;
-- 8 Profit vs loss count
SELECT Profit_Flag, COUNT(*) AS total FROM marketing_campaigns GROUP BY Profit_Flag;
-- 9 Customer segment revenue
SELECT Customer_Segment, SUM(Revenue) AS total_revenue FROM marketing_campaigns GROUP BY Customer_Segment ORDER BY total_revenue DESC;
-- 10 Language wise ROI
SELECT Language, AVG(ROI) AS avg_roi FROM marketing_campaigns GROUP BY Language ORDER BY avg_roi DESC;
-- 11 Monthly revenue
SELECT Year, Month, SUM(Revenue) AS total_revenue FROM marketing_campaigns GROUP BY Year, Month ORDER BY Year, Month;
-- 12 Average engagement by brand
SELECT Brand, AVG(Engagement_Score) AS avg_engagement FROM marketing_campaigns GROUP BY Brand;
-- 13 Conversion rate
SELECT Brand, AVG(Conversions / NULLIF(Clicks,0)) AS conversion_rate FROM marketing_campaigns GROUP BY Brand;
-- 14 Click through rate
SELECT Brand, AVG(Clicks / NULLIF(Impressions,0)) AS ctr FROM marketing_campaigns GROUP BY Brand;
-- 15 Cost per conversion
SELECT Brand, AVG(Acquisition_Cost / NULLIF(Conversions,0)) AS cost_per_conversion FROM marketing_campaigns GROUP BY Brand;
-- 16 Best campaign type per brand
SELECT Brand, Campaign_Type, AVG(ROI) AS avg_roi FROM marketing_campaigns GROUP BY Brand, Campaign_Type ORDER BY Brand, avg_roi DESC;
-- 17 Revenue above average
SELECT * FROM marketing_campaigns WHERE Revenue > (SELECT AVG(Revenue) FROM marketing_campaigns);
-- 18 High ROI campaigns
SELECT Campaign_ID, Brand, ROI FROM marketing_campaigns WHERE ROI > 5 ORDER BY ROI DESC;
-- 19 Lead to conversion ratio
SELECT Brand, AVG(Conversions / NULLIF(Leads,0)) AS lead_conversion_ratio FROM marketing_campaigns GROUP BY Brand;
-- 20 Target audience performance
SELECT Target_Audience, SUM(Revenue) AS revenue, AVG(ROI) AS avg_roi FROM marketing_campaigns GROUP BY Target_Audience ORDER BY revenue DESC;
-- 21 Create revenue view
CREATE OR REPLACE VIEW brand_revenue_view AS SELECT Brand, SUM(Revenue) AS total_revenue FROM marketing_campaigns GROUP BY Brand;
-- 22 Use view
SELECT * FROM brand_revenue_view;
-- 23 Ranking campaigns by revenue
SELECT Campaign_ID, Brand, Revenue, RANK() OVER(ORDER BY Revenue DESC) AS revenue_rank FROM marketing_campaigns LIMIT 20;
-- 24 Brand revenue ranking
SELECT Brand, SUM(Revenue) AS revenue, RANK() OVER(ORDER BY SUM(Revenue) DESC) AS brand_rank FROM marketing_campaigns GROUP BY Brand;
-- 25 Running monthly revenue
SELECT Year, Month, SUM(Revenue) AS monthly_revenue, SUM(SUM(Revenue)) OVER(ORDER BY Year, Month) AS running_revenue FROM marketing_campaigns GROUP BY Year, Month;
-- 26 Campaign count by channel
SELECT Channel_Used, COUNT(*) AS total FROM marketing_campaigns GROUP BY Channel_Used ORDER BY total DESC LIMIT 10;
-- 27 Max revenue by brand
SELECT Brand, MAX(Revenue) AS max_revenue FROM marketing_campaigns GROUP BY Brand;
-- 28 Min acquisition cost by brand
SELECT Brand, MIN(Acquisition_Cost) AS min_cost FROM marketing_campaigns GROUP BY Brand;
-- 29 Average duration by campaign type
SELECT Campaign_Type, AVG(Duration) AS avg_duration FROM marketing_campaigns GROUP BY Campaign_Type;
-- 30 Profitable revenue by brand
SELECT Brand, SUM(Revenue) AS profitable_revenue FROM marketing_campaigns WHERE Profit_Flag = 1 GROUP BY Brand;
