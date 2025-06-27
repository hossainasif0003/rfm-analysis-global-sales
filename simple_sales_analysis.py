# ====================================================
# Simple Sales Analysis
# ----------------------------------------------------
# Dataset: sales_data.csv
# Author: Asif Hossain
# Description:
# This script performs an exploratory sales analysis
# covering revenue trends, profit distribution, and 
# visual insights across countries and time.
# Libraries used: pandas, matplotlib, seaborn
# ====================================================

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file = r"C:\Users\Asif Hossain\Desktop\pyhon files\sales_data.csv"
df = pd.read_csv(file)

# Preview the first few rows
df.head()

# Convert the 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Create a new column for profit margin (rounded to 2 decimal places)
df["Profit_margin"] = (df["Profit"] / df["Revenue"] * 100).round(2)

# Display the new column
print(df[["Profit", "Revenue", "Profit_margin"]].head())

# ====================================================
# 1. Top 5 Products by Revenue
# ====================================================
product_revenue = df.groupby("Product")[["Revenue"]].sum().sort_values(by="Revenue", ascending=False).head(5)
print("Top 5 Products by Revenue:\n", product_revenue)

# ====================================================
# 2. Monthly Revenue Trend
# ====================================================
# Ensure 'Month' column is ordered correctly
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)

# Group by Month
monthly_revenue = df.groupby("Month", as_index=False)["Revenue"].sum()

# Line plot for monthly revenue trend
plt.style.use('dark_background')
plt.figure(figsize=(10, 5))
plt.plot(monthly_revenue["Month"], monthly_revenue["Revenue"], marker="x", color="teal")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.tight_layout()
# plt.savefig("monthly_revenue_trend.png", dpi=300)  # Optional
plt.show()

# ====================================================
# 3. Average Profit by Country
# ====================================================
avg_profit_country = df.groupby("Country", as_index=False)["Profit"].mean().round(2)
print("Average Profit by Country:\n", avg_profit_country)

# Bar chart of average profit per country
plt.figure(figsize=(12, 6))
plt.bar(avg_profit_country["Country"], avg_profit_country["Profit"], color='skyblue')
plt.title("Average Profit by Country")
plt.xlabel("Country")
plt.ylabel("Average Profit")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("avg_profit_by_country.png", dpi=300)  # Optional
plt.show()

# ====================================================
# 4. Histogram of Revenue
# ====================================================
plt.figure(figsize=(10, 5))
sns.histplot(df["Revenue"], bins=50, color='purple')
plt.title("Distribution of Revenue", fontsize=18, color='white')
plt.xlabel("Revenue", fontsize=14, color='white')
plt.ylabel("Frequency", fontsize=14, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
# plt.savefig("revenue_distribution.png", dpi=300)  # Optional
plt.show()

# ====================================================
# 5. Scatter Plot: Revenue vs Profit by Country
# ====================================================
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Revenue", y="Profit", hue="Country", palette="Set2")
plt.title("Sales vs Profit by Country", fontsize=18, color='white')
plt.xlabel("Revenue (Sales)", fontsize=14, color='white')
plt.ylabel("Profit", fontsize=14, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.legend(title="Country", fontsize=10, title_fontsize=12)
plt.tight_layout()
# plt.savefig("revenue_vs_profit.png", dpi=300)  # Optional
plt.show()
