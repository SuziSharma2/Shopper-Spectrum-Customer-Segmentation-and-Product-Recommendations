# Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations

This project explores real-world e-commerce transaction data to identify customer behavior patterns and deliver personalized product recommendations. Using RFM-based clustering and collaborative filtering, we help businesses improve targeting, retention, and engagement strategies through data-driven insights.

---

## 📌 Problem Statement

The global e-commerce industry produces massive volumes of transactional data. However, most platforms struggle to convert this data into actionable insights.

This project addresses two key problems:

- 🎯 Segmenting customers into meaningful groups using **Recency, Frequency, and Monetary (RFM)** values via **KMeans Clustering**
- 🛍️ Recommending relevant products using **item-based collaborative filtering** with **cosine similarity**

---

## 📊 Dataset Description

The dataset contains invoice-level transaction data from a UK-based online retail store (2022–2023). Key columns include:

| Column        | Description                            |
|---------------|----------------------------------------|
| InvoiceNo     | Unique invoice number                  |
| StockCode     | Unique product ID                      |
| Description   | Product name                           |
| Quantity      | Quantity purchased                     |
| InvoiceDate   | Date and time of transaction           |
| UnitPrice     | Price per item                         |
| CustomerID    | Unique customer identifier             |
| Country       | Customer's country                     |

---

## 🚀 Project Workflow

### 1️⃣ Data Preprocessing
- Removed missing `CustomerID` values
- Excluded cancelled invoices
- Removed rows with negative or zero `Quantity` or `UnitPrice`
- Created `TotalAmount = Quantity * UnitPrice`

### 2️⃣ Exploratory Data Analysis (EDA)
- Transaction volume by country
- Top-selling products
- Monthly sales trends
- RFM distributions and correlations

### 3️⃣ RFM Feature Engineering
- **Recency** = Days since last purchase
- **Frequency** = Number of transactions
- **Monetary** = Total spend
- Normalized features using StandardScaler

### 4️⃣ Clustering for Customer Segmentation
- Used **KMeans Clustering**
- Selected optimal clusters with **Elbow Method** and **Silhouette Score**
- Labeled segments: **High-Value**, **Regular**, **Occasional**, **At-Risk**

### 5️⃣ Product Recommendation Engine
- Built a **CustomerID–StockCode** matrix
- Calculated **cosine similarity** between products
- Recommended top 5 similar products

---

## 🧠 Technologies Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**, **Joblib**
- **Matplotlib**, **Seaborn**
- **Streamlit** (for dashboard deployment)

---

## 🌐 Streamlit Web App Features

### 1️⃣ Product Recommender
- Input: Product ID or Name
- Output: 5 similar products based on past purchases

### 2️⃣ Customer Segment Predictor
- Input: Recency, Frequency, Monetary values
- Output: Customer segment (e.g., High-Value, At-Risk)

---

## 📂 Project Structure
- shopper-spectrum/
- │
- ├── data/
- │   └── online_retail.csv                ← raw input data
- │
- ├── notebooks/
- │   └── shopper_spectrum_eda.ipynb       ← full EDA + clustering + recommendation
- │
- ├── app/
- │   └── app.py                           ← Streamlit UI 
- │
- ├── models/
- │   ├── kmeans_rfm_model.pkl             ← from notebook
- │   ├── rfm_scaler.pkl                   ← from notebook
- │   └── product_similarity.pkl           ← from notebook
- │
- └── README.md                            ← project summary



