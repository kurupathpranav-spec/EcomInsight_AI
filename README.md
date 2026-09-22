# 🛒 EcomInsight AI

**EcomInsight AI** is an interactive e-commerce analytics dashboard built with **Streamlit**, analysing the **Madhav E-Commerce Sales Dataset** from Kaggle. It provides deep insights into sales performance, category trends, geographic distribution, payment behaviour, and a machine-learning–based sales forecast — all from a single, self-contained Python application.

> **Submitted for:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
> **Author:** Pranav Kurupath

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `Orders.csv` | Order-level information — Order ID, Order Date, Customer Name, State, City |
| `Details.csv` | Transaction-level information — Order ID, Amount, Profit, Quantity, Category, Sub-Category, Payment Mode |

Both files are joined on **Order ID** to produce the unified analytical dataset.

**Original Kaggle Dataset:**
🔗 [https://www.kaggle.com/datasets/amitkumar209/madhav-e-commerce-sales-dataset](https://www.kaggle.com/datasets/amitkumar209/madhav-e-commerce-sales-dataset)

---

## 🚀 Quick Start

### 1 · Clone / download the project
```bash
git clone https://github.com/kurupathpranav-spec/EcomInsight_AI.git
```

### 2 · Install dependencies
```bash
pip install -r requirements.txt
```

### 3 · Place the dataset files
Ensure the `dataset/` folder (containing `Orders.csv` and `Details.csv`) is present in the project root — it is included in the repository by default.

### 4 · Run the Streamlit app
```bash
streamlit run PranavKurupath_EcomInsightAI.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
EcomInsight_AI/
├── .streamlit/
│   └── config.toml
├── dataset/
│   ├── Orders.csv
│   └── Details.csv
├── PranavKurupath_EcomInsightAI.py
├── PranavKurupath_EcomInsightAI_ProjectReport.docx
├── README.md
└── requirements.txt

- `.streamlit/config.toml` – Streamlit theme and application configuration
- `dataset/` – Contains the Orders and Details CSV files
- `PranavKurupath_EcomInsightAI.py` – Main Streamlit application
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation
- `PranavKurupath_EcomInsightAI_ProjectReport.docx` – Detailed project report
```

---

## ✨ Features

### 📊 Dashboard & KPIs
- Total Sales, Total Profit, Profit Margin %, Total Orders, Average Order Value, Units Sold
- Sidebar filters: State, Category, Payment Mode, Date Range
- All charts respond to active filter selections

### 📈 Sales Trends
- Monthly Sales & Profit trend lines
- Quarterly sales bar chart
- Monthly Profit Margin % trend
- Top 10 customers by sales

### 🗂️ Category Analysis
- Sales share pie chart (Category)
- Sales vs Profit grouped bar chart
- Top 20 sub-categories by sales
- Per-sub-category profit bar chart (red = loss, green = profit)

### 🗺️ Geographic Analysis
- State-wise sales bar chart (colour-coded by profit)
- Top 8 states — sales share pie chart
- Top 15 cities by sales
- State-level summary table

### 💳 Payment Mode Analysis
- Sales share by payment mode (pie)
- Sales vs Profit by payment mode (bar)
- Category × Payment Mode heatmap
- Detailed payment mode table

### 🔮 Sales Forecasting
- Linear Regression model trained on monthly aggregated sales
- 1–6 month adjustable forecast horizon
- Historical vs Forecast line chart
- R² score displayed for transparency

### 🤖 AI Business Analyst
- Natural-language question interface
- Quick-select example questions
- Answers grounded in actual dataset figures
- Topics: revenue, profit, categories, states, cities, payment modes, trends, forecasts, recommendations

### 💡 AI Insights & Recommendations
- Automatically generated insights: margin health, top/worst performers, trend direction, loss-making sub-categories
- Risk/warning flags for high-sales low-margin categories
- Actionable business recommendations

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly Express, Plotly Graph Objects |
| Machine Learning | scikit-learn (LinearRegression) |
| Report Generation | python-docx |
| Language | Python 3.10+ |

---

## 📋 Deliverables

| File | Purpose |
|------|---------|
| `PranavKurupath_EcomInsightAI.py` | Complete Streamlit application |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `PranavKurupath_EcomInsightAI_ProjectReport.docx` | Formal project report |

---

## 📄 License

This project is submitted as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026** and is intended for educational and evaluation purposes.
