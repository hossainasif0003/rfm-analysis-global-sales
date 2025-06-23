# Online Retail Data Analysis with RFM Segmentation

This is a simple Python project that looks at online shopping data. It helps understand which products and countries bring in the most money, and it finds the best customers using RFM (Recency, Frequency, Monetary) analysis.

##  Author
Asif Hossain

##  What This Project Does

- Reads a file called **Online Retail.csv**.
- Cleans the data (removes empty or wrong data).
- Shows:
  - Top 10 products that earned the most money.
  - Countries that spent the most.
  - How much money was made each month.
- Finds how much each customer spends.
- Groups customers into segments using:
  - **Recency** (how recently they bought something)
  - **Frequency** (how many times they bought)
  - **Monetary** (how much money they spent)
- Shows the best customers with score `101010`.

##  What Tools Are Used

- **Pandas** – to read and work with data
- **Matplotlib** – to make graphs
- **Seaborn** – to make nicer graphs

##  How to Run It

1. Make sure Python is installed.
2. Install the needed tools:
   ```bash
   pip install pandas matplotlib seaborn
