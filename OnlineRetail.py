import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Load Dataset
# ------------------------------__=r"C:\Users\Asif Hossain\Desktop\Dataset\Python Cheat Sheet\Online Retail\Online Retail.csv"
file=r"C:\Users\Asif Hossain\Desktop\Dataset\Online Retail Visualisation\Online Retail.csv"
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
df = df.dropna(subset=['CustomerID', 'Description'])  # Drop rows with missing critical info
df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]  # Remove rows with invalid values

# ------------------------------
# Step 4: Feature Engineering
# ------------------------------
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# ------------------------------
# Step 5: Top Products by Revenue
# ------------------------------
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
monthly_revenue = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
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

# Save the plot
plt.savefig('monthly_revenue.png', dpi=300, bbox_inches='tight')
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
numeric_df = df.select_dtypes(include='number').drop(columns=['CustomerID', 'Year', 'Month'])
correlation_matrix = numeric_df.corr()

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

# Save the plot
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------
# Step 10: Customer Segmentation Using RFM
# ------------------------------
latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
recency_df = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
recency_df['Recency'] = (latest_date - recency_df['InvoiceDate']).dt.days
recency_df = recency_df[['CustomerID', 'Recency']]

frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency_df.rename(columns={"InvoiceNo": "Frequency"}, inplace=True)

monetary_df = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace=True)

rfm = recency_df.merge(frequency_df, on='CustomerID').merge(monetary_df, on='CustomerID')
rfm['R_Score'] = pd.qcut(rfm['Recency'], 10, labels=range(10, 0, -1)).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 10, labels=range(1, 11)).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 10, labels=range(1, 11)).astype(int)
rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

best_customers = rfm[rfm['RFM_Score'] == '101010']
print("\nBest Customers (RFM Score = 101010):")
print(best_customers)

# ------------------------------
# Step 11: Visualize RFM Distributions
# ------------------------------
sns.histplot(rfm['Recency'], bins=20, kde=True)
plt.title("Recency Distribution")
# Save the plot
plt.savefig('recency_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

sns.histplot(rfm['Frequency'], bins=20, kde=True)
plt.title("Frequency Distribution")
# Save the plot
plt.savefig('frequency_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

sns.histplot(rfm['Monetary'], bins=20, kde=True)
plt.title("Monetary Distribution")
# Save the plot
plt.savefig('monetary_distribution.png', dpi=300, bbox_inches='tight')
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

# Save the plot
plt.savefig('top_5_countries_by_revenue.png', dpi=300, bbox_inches='tight')
plt.show()
