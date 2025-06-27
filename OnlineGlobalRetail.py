# Author: Asif Hossain
# Project: Online Retail Data Analysis with RFM Segmentation
# Description: This script analyzes online retail sales data to identify
#              top products, countries, monthly revenue trends, and
#              segments customers based on Recency, Frequency, and Monetary (RFM) values.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
file = r"C:\Users\Asif Hossain\Desktop\Python Cheat Sheet\Online Retail.csv"
# Read CSV using 'latin1' encoding to avoid decoding errors from special characters
df = pd.read_csv(file, encoding='latin1')

# ------------------------------
# Step 2: Initial Data Exploration
# ------------------------------
print("Initial Data Preview:")
print(df.head())  # Show first 5 rows to understand data structure

print("\nDataset Info:")
df.info()  # Summary of columns, data types, and non-null counts

print("\nStatistical Summary:")
df.describe()  # Basic stats (mean, std, min, max) for numeric columns

# ------------------------------
# Step 3: Data Cleaning
# ------------------------------
df = df.drop_duplicates()  # Remove exact duplicate rows to avoid double counting

# Drop rows with missing critical info: CustomerID or Description (product name)
df = df.dropna(subset=['CustomerID', 'Description'])

# Remove rows where UnitPrice or Quantity are zero or negative, as these don't make sense for revenue calculations
df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]

# ------------------------------
# Step 4: Feature Engineering
# ------------------------------
# Calculate total price for each transaction (Quantity * UnitPrice)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Convert InvoiceDate to datetime format for easy date operations
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract Year and Month as separate columns for time series analysis
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# ------------------------------
# Step 5: Top Products by Revenue
# ------------------------------
# Group data by product description and sum their total sales revenue
top_products = (
    df.groupby("Description")[["TotalPrice"]]
    .sum()
    .sort_values("TotalPrice", ascending=False)
    .head(10)  # Get top 10 products
)
print("\nTop 10 Products by Revenue:")
print(top_products)

# ------------------------------
# Step 6: Monthly Revenue Trends
# ------------------------------
# Group by Year and Month, then sum revenue to analyze monthly trends
monthly_revenue = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()

# Plot monthly revenue over time with a dark-themed chart
plt.figure(figsize=(10, 6), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

plt.plot(monthly_revenue['Month'], monthly_revenue['TotalPrice'], marker='o', color='white')
plt.xlabel('Month', color='white', fontsize=16)
plt.ylabel('Total Revenue', color='white', fontsize=16)
plt.title('Monthly Revenue Over Time', color='white', fontsize=18)
plt.xticks(color='white', fontsize=14)
plt.yticks(color='white', fontsize=14)
plt.grid(True, color='gray')
plt.tight_layout()
plt.show()

# ------------------------------
# Step 7: Top Countries by Revenue
# ------------------------------
country_revenue = (
    df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\nTop Countries by Revenue:")
print(country_revenue.head(10))

# ------------------------------
# Step 8: Average Revenue per Customer
# ------------------------------
avg_revenue = df.groupby("CustomerID")[["TotalPrice"]].mean()
avg_revenue.index = avg_revenue.index.astype(int)  # Convert CustomerID to int for clarity

print("\nAverage Revenue per Customer:")
print(avg_revenue.head())

# ------------------------------
# Step 9: Correlation Analysis
# ------------------------------
# Select numeric columns only, drop identifiers and date parts to focus on meaningful metrics
numeric_df = df.select_dtypes(include='number').drop(columns=['CustomerID', 'Year', 'Month'])

correlation_matrix = numeric_df.corr()

# Plot heatmap of correlations with a colorful palette for easy visualization
plt.figure(figsize=(6, 5), facecolor='black')
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='gist_rainbow',
    linewidths=0.5,
    cbar=True,
    annot_kws={"size": 12, "color": 'black'},
    fmt=".2f"
)
plt.title("Correlation Heatmap", color='white', fontsize=18)
plt.xticks(color='white', fontsize=13, rotation=20)
plt.yticks(color='white', fontsize=13, rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------
# Step 10: Customer Segmentation Using RFM
# ------------------------------
# Reference date for recency calculation is one day after the latest invoice date
latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Recency: Days since last purchase for each customer
recency_df = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
recency_df['Recency'] = (latest_date - recency_df['InvoiceDate']).dt.days
recency_df = recency_df[['CustomerID', 'Recency']]

# Frequency: Number of unique invoices (purchases) per customer
frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency_df.rename(columns={"InvoiceNo": "Frequency"}, inplace=True)

# Monetary: Total money spent by each customer
monetary_df = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace=True)

# Combine Recency, Frequency, and Monetary data into one DataFrame
rfm = recency_df.merge(frequency_df, on='CustomerID').merge(monetary_df, on='CustomerID')

# Score each metric into 10 quantile-based bins, where 10 is best score
rfm['R_Score'] = pd.qcut(rfm['Recency'], 10, labels=range(10, 0, -1)).astype(int)  # Lower recency = better
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 10, labels=range(1, 11)).astype(int)

# Concatenate R, F, M scores to form combined RFM score
rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

# Filter customers with best possible RFM score '101010' (lowest recency, highest frequency and monetary)
best_customers = rfm[rfm['RFM_Score'] == '101010']
print("\nBest Customers (RFM Score = 101010):")
print(best_customers)

# ------------------------------
# Step 11: Visualize RFM Distributions
# ------------------------------
sns.histplot(rfm['Recency'], bins=20, kde=True)
plt.title("Recency Distribution")
plt.show()

sns.histplot(rfm['Frequency'], bins=20, kde=True)
plt.title("Frequency Distribution")
plt.show()

sns.histplot(rfm['Monetary'], bins=20, kde=True)
plt.title("Monetary Distribution")
plt.show()

# ------------------------------
# Step 12: Visualize Top 5 Countries by Revenue
# ------------------------------
top5_countries = country_revenue.head(5)
sns.barplot(x="Country", y="TotalPrice", data=top5_countries)
plt.xlabel("Country", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.title("Top 5 Countries by Total Revenue", fontsize=18)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
