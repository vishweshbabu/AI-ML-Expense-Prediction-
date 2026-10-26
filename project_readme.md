# AI Personal Finance and Smart Expense Predictor
**Project 03 | Team 03**  
**Technology Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Jupyter Notebook, Streamlit

---

## 1. Project Overview
The **AI Personal Finance and Smart Expense Predictor** is a personal finance assistant that organizes income and expenses, summarizes spending patterns, and estimates future monthly expenses from historical transaction records. 

The tool is designed to help users understand their financial habits, compare monthly spending against category budgets, and anticipate likely month-end expenses. **It does not provide investment advice or guarantee future spending outcomes.**

---

## 2. Team Allocation & Roles

| S. No. | Team Member | Assigned Role | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **1** | **Mitul Jain** | Data Engineering & Preprocessing | Clean 4-year synthetic dataset, preprocessing notebook, and data dictionary |
| **2** | **Vishwesh Babu T J** *(with Mitul Jain)* | **AI/ML & Expense Prediction** | **ML/forecasting module, model evaluation report, and data-driven financial insights** |
| **3** | **Asmi Zilpilwar** | Financial Analytics & EDA | Exploratory data analysis notebook, financial summaries, and category charts |
| **4** | **Kaushal Shrikant Jadhav** | Frontend & Dashboard Development | Streamlit interactive dashboard, metric cards, and prediction integration |
| **5** | **Sweta Koshta** | Documentation, Testing & QA | Project documentation, privacy verification, acceptance testing, and final report |

---

## 3. AI/ML & Expense Prediction Module (Vishwesh Babu T J)

### 3.1 How the Forecasting Engine Works
1. **Validation & Insufficient-History Safeguard:**  
   Before running any machine learning model, the pipeline verifies whether the dataset contains at least $4$ months of historical records. If the transaction history is empty or has $< 4$ months of data, the system suppresses ML predictions and displays a clear warning to prevent **false precision**.
2. **Feature Engineering & Mid-Month Estimation:**  
   Daily transactions are aggregated by `YearMonth` and `Category` (`Food`, `Rent`, `Shopping`, `Transport`, `Utilities`). The module computes strictly historical lag features:
   * Previous 1-month spend (`Prev_1M_Spend`)
   * Previous 2-month spend (`Prev_2M_Spend`)
   * 3-month rolling average (`Prev_3M_Rolling_Avg`)
   * First-half month spending (`First_Half_Spend`, Days $1\text{--}15$) to estimate likely month-end totals from partial-month activity
   * Temporal indicators (`Month_Num`, `Year_Num`, `Category_Code`)
3. **Time-Aware Train/Test Split (Zero Data Leakage):**  
   To prevent random data leakage across time, the 4-year synthetic dataset (`2022-01` to `2025-12`) is split strictly in chronological order:
   * **Training Period:** April 2022 – June 2025 ($39$ months)
   * **Held-Out Evaluation Period:** July 2025 – December 2025 ($6$ months)
4. **Models Compared:**
   * **Baseline Model:** 3-Month Historical Moving Average
   * **Linear Regression:** Ordinary Least Squares regression on historical lags and seasonality
   * **Random Forest Regressor:** Non-linear ensemble model (`n_estimators=150`, `max_depth=8`, `random_state=42`)

---

### 3.2 Model Evaluation Summary (Held-Out Test Set: Jul–Dec 2025)
Models are evaluated using **Mean Absolute Error ($\text{MAE}$)**:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_{\text{actual}, i} - \hat{y}_{\text{predicted}, i} \right|$$

| Model | Approach | Overall $\text{MAE}$ (INR) | Comparison vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Baseline Forecast** | 3-Month Moving Average | ₹1,170.12 | Baseline Reference |
| **Linear Regression** | Lag + Temporal OLS | ₹1,081.12 | $7.61\%$ lower error |
| **Random Forest Regressor** | Tree Ensemble | **₹1,050.02** | **$10.26\%$ lower error (Best Overall)** |

#### Category-Wise $\text{MAE}$ Breakdown (INR)
| Category | Baseline $\text{MAE}$ | Linear Regression $\text{MAE}$ | Random Forest $\text{MAE}$ | Best Performing Model |
| :--- | :--- | :--- | :--- | :--- |
| **Food** | ₹1,590.05 | **₹1,222.03** | ₹1,401.92 | Linear Regression |
| **Rent** | **₹368.35** | ₹552.50 | ₹604.30 | Baseline (3M Moving Avg) |
| **Shopping** | ₹2,908.94 | **₹2,691.93** | ₹2,701.73 | Linear Regression |
| **Transport** | ₹397.81 | ₹466.57 | **₹222.56** | Random Forest |
| **Utilities** | ₹585.46 | ₹472.57 | **₹319.58** | Random Forest |

---

### 3.3 Automated Financial Insights & Anomaly Detection
* **Unusual Spending Detection:** Computes a 6-month rolling mean ($\mu_{6\text{M}}$) and standard deviation ($\sigma_{6\text{M}}$) per category. Any month where the $Z$-score satisfies:
  $$Z = \frac{\text{Amount} - \mu_{6\text{M}}}{\sigma_{6\text{M}}} > 2.0$$
  is automatically flagged as an **Unusual Spending Alert** (e.g., Festival Shopping spike in Nov 2025 at $Z = 5.26$, and Food surge in Oct 2025 at $Z = 9.83$).
* **Monthly Category Budget Alerts:** Compares actual and forecasted category totals against user-defined monthly category budgets and triggers overspending alerts when limits are exceeded.

---

## 4. Repository Structure

```text
├── vishwesh_expense_predictor_module.py          # Core AI/ML Forecasting & Insights Engine
├── Vishwesh_AIML_Expense_Prediction_Notebook.ipynb # Jupyter Notebook for Training, Evaluation & Insights
├── Vishwesh_Model_Evaluation_and_Insights_Report.pdf # Formal Evaluation & Insights Report
├── cleaned_transactions.csv                      # 4-Year Synthetic Transaction Dataset
├── app.py                                        # Streamlit Frontend Dashboard
└── README.md                                     # Project Documentation, Assumptions & Privacy Note
```

---

## 5. Setup & Installation Instructions

### Prerequisites
Ensure you have **Python 3.9+** installed. Install the required packages using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter
```

### Running the AI/ML Module Standalone
You can test the forecasting and anomaly detection module directly in Python or Jupyter Notebook:

```python
import pandas as pd
from vishwesh_expense_predictor_module import SmartExpensePredictor

# 1. Load the synthetic dataset
df = pd.read_csv("cleaned_transactions.csv")

# 2. Initialize predictor and validate historical records
predictor = SmartExpensePredictor(currency_symbol="₹")
prep = predictor.validate_and_prepare(df)

if prep["status"] == "ok":
    monthly_df = prep["data"]
    
    # 3. Train models and view time-based MAE evaluation
    eval_report = predictor.train_and_evaluate(monthly_df, test_months=6)
    print("Overall MAE:", eval_report["Overall_MAE"])
    
    # 4. Predict next month's category expenses
    forecasts = predictor.predict_next_month_expenses(monthly_df)
    print(forecasts)
    
    # 5. Generate unusual spending & budget insights
    budgets = {"Food": 8000, "Shopping": 6000, "Transport": 4000, "Rent": 19000, "Utilities": 3500}
    insights = predictor.generate_financial_insights(monthly_df, category_budgets=budgets)
    for alert in insights:
        print(alert["Message"])
else:
    print(prep["message"])
```

### Running the Streamlit Dashboard Demo
```bash
streamlit run app.py
```

---

## 6. Forecast Assumptions & Limitations

1. **Statistical Estimates Only:** All model outputs are explicitly labelled as **`Estimated Spend (Not Guaranteed)`**. They represent statistical projections derived from past records and must not be interpreted as financial, legal, or investment advice.
2. **Historical Continuity Assumption:** Both Linear Regression and Random Forest assume that historical spending habits, income stability, and seasonal patterns will continue into the near future.
3. **Unforeseen Events:** The models cannot anticipate sudden external shocks, medical emergencies, relocation, inflation spikes, or unrecorded cash transactions.
4. **Cold-Start / Insufficient History Limitation:** Reliable time-series lag calculation requires at least $4$ months of consistent transaction history. When fewer than $4$ months are available, forecasting is automatically disabled to avoid misleading the user with false precision.

---

## 7. Data Privacy & Security Note

* **Synthetic Data Only:** In strict compliance with project guidelines, this application is developed and demonstrated exclusively using a **synthetic transaction dataset**.
* **Zero Sensitive Credentials:** The system **does not** connect to real bank accounts and **never** collects, processes, or stores bank login credentials, account numbers, UPI PINs, or payment card details.
* **Local Execution:** All data cleaning, exploratory analysis, and machine learning inference run locally within the user's environment.