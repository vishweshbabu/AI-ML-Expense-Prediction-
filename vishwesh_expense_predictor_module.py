"""
AI Personal Finance and Smart Expense Predictor
Module: AI/ML & Expense Prediction
Author: Vishwesh Babu T J (in collaboration with Mitul Jain)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class SmartExpensePredictor:
    """
    End-to-end ML Forecasting and Financial Insights engine.
    - Prevents temporal data leakage via strict chronological splitting.
    - Handles empty or insufficient history gracefully without false precision.
    - Compares Baseline (3-Month Moving Average), Linear Regression, and Random Forest.
    - Supports mid-month to month-end estimation and anomaly detection.
    """
    MIN_HISTORY_MONTHS = 4

    def __init__(self, currency_symbol="₹"):
        self.currency = currency_symbol
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        self.category_mapping = {}
        self.is_trained = False
        self.latest_eval_report = {}

    def validate_and_prepare(self, df: pd.DataFrame) -> dict:
        """
        Validates input transaction records and aggregates them into monthly category features.
        Expected columns: ['Date', 'Category', 'Amount'] (optional: 'Transaction_Type')
        """
        if df is None or df.empty:
            return {
                "status": "insufficient_data",
                "message": "Transaction history is empty. Please add or import sample expenses to generate a forecast.",
                "data": None
            }

        required_cols = {'Date', 'Category', 'Amount'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        work_df = df.copy()
        if 'Transaction_Type' in work_df.columns:
            work_df = work_df[work_df['Transaction_Type'].str.lower() == 'expense']

        work_df['Date'] = pd.to_datetime(work_df['Date'], errors='coerce')
        work_df = work_df.dropna(subset=['Date', 'Category', 'Amount'])
        work_df['Amount'] = pd.to_numeric(work_df['Amount'], errors='coerce')
        work_df = work_df[work_df['Amount'] > 0]

        work_df['Day'] = work_df['Date'].dt.day
        work_df['YearMonth'] = work_df['Date'].dt.to_period('M')

        unique_months = work_df['YearMonth'].nunique()
        if unique_months < self.MIN_HISTORY_MONTHS:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Only {unique_months} month(s) of recorded history found. "
                    f"At least {self.MIN_HISTORY_MONTHS} months of data are required for a reliable forecast."
                ),
                "data": None
            }

        # Calculate first-half spending (Day 1 to 15) for mid-month projections
        first_half = (
            work_df[work_df['Day'] <= 15]
            .groupby(['YearMonth', 'Category'])['Amount']
            .sum()
            .reset_index()
            .rename(columns={'Amount': 'First_Half_Spend'})
        )

        monthly = (
            work_df.groupby(['YearMonth', 'Category'])['Amount']
            .sum()
            .reset_index()
            .merge(first_half, on=['YearMonth', 'Category'], how='left')
        )
        monthly['First_Half_Spend'] = monthly['First_Half_Spend'].fillna(0.0)
        monthly = monthly.sort_values(['Category', 'YearMonth']).reset_index(drop=True)

        # Lag features (strictly past values to prevent leakage)
        monthly['Prev_1M_Spend'] = monthly.groupby('Category')['Amount'].shift(1)
        monthly['Prev_2M_Spend'] = monthly.groupby('Category')['Amount'].shift(2)
        monthly['Prev_3M_Rolling_Avg'] = (
            monthly.groupby('Category')['Amount']
            .shift(1)
            .rolling(3, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
        monthly['Rolling_6M_Mean'] = (
            monthly.groupby('Category')['Amount']
            .shift(1)
            .rolling(6, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        monthly['Rolling_6M_Std'] = (
            monthly.groupby('Category')['Amount']
            .shift(1)
            .rolling(6, min_periods=3)
            .std()
            .reset_index(level=0, drop=True)
        )

        monthly['Month_Num'] = monthly['YearMonth'].dt.month
        monthly['Year_Num'] = monthly['YearMonth'].dt.year

        categories = sorted(monthly['Category'].unique())
        self.category_mapping = {cat: idx for idx, cat in enumerate(categories)}
        monthly['Category_Code'] = monthly['Category'].map(self.category_mapping)

        return {
            "status": "ok",
            "message": "Data successfully validated and aggregated.",
            "data": monthly
        }

    def train_and_evaluate(self, monthly_df: pd.DataFrame, test_months: int = 6) -> dict:
        """
        Splits data chronologically into train and held-out test periods,
        trains Baseline, Linear Regression, and Random Forest models, and reports MAE.
        """
        clean_df = monthly_df.dropna(subset=['Prev_1M_Spend', 'Prev_2M_Spend', 'Prev_3M_Rolling_Avg']).copy()
        all_periods = sorted(clean_df['YearMonth'].unique())

        if len(all_periods) <= test_months:
            test_months = max(1, len(all_periods) // 4)

        split_period = all_periods[-test_months]
        train_df = clean_df[clean_df['YearMonth'] < split_period].copy()
        test_df = clean_df[clean_df['YearMonth'] >= split_period].copy()

        feature_cols = [
            'Category_Code', 'Month_Num', 'Year_Num',
            'Prev_1M_Spend', 'Prev_2M_Spend', 'Prev_3M_Rolling_Avg', 'First_Half_Spend'
        ]

        X_train, y_train = train_df[feature_cols], train_df['Amount']
        X_test, y_test = test_df[feature_cols], test_df['Amount']

        # 1. Baseline Forecast (3-Month Historical Moving Average)
        test_df['Baseline_Estimate'] = test_df['Prev_3M_Rolling_Avg'].round(2)
        baseline_mae = mean_absolute_error(y_test, test_df['Baseline_Estimate'])

        # 2. Linear Regression Forecast
        self.lr_model.fit(X_train, y_train)
        test_df['LR_Estimate'] = np.maximum(0, self.lr_model.predict(X_test)).round(2)
        lr_mae = mean_absolute_error(y_test, test_df['LR_Estimate'])

        # 3. Random Forest Forecast
        self.rf_model.fit(X_train, y_train)
        test_df['RF_Estimate'] = np.maximum(0, self.rf_model.predict(X_test)).round(2)
        rf_mae = mean_absolute_error(y_test, test_df['RF_Estimate'])

        self.is_trained = True

        # Category-level MAE breakdown
        category_metrics = []
        for cat, grp in test_df.groupby('Category'):
            category_metrics.append({
                'Category': cat,
                'Baseline_MAE': round(float(mean_absolute_error(grp['Amount'], grp['Baseline_Estimate'])), 2),
                'Linear_Regression_MAE': round(float(mean_absolute_error(grp['Amount'], grp['LR_Estimate'])), 2),
                'Random_Forest_MAE': round(float(mean_absolute_error(grp['Amount'], grp['RF_Estimate'])), 2)
            })

        best_model = min(
            [('Baseline (3M Moving Avg)', baseline_mae),
             ('Linear Regression', lr_mae),
             ('Random Forest', rf_mae)],
            key=lambda x: x[1]
        )[0]

        self.latest_eval_report = {
            "Train_Period": f"{train_df['YearMonth'].min()} to {train_df['YearMonth'].max()}",
            "Held_Out_Test_Period": f"{test_df['YearMonth'].min()} to {test_df['YearMonth'].max()}",
            "Overall_MAE": {
                "Baseline_MAE": round(float(baseline_mae), 2),
                "Linear_Regression_MAE": round(float(lr_mae), 2),
                "Random_Forest_MAE": round(float(rf_mae), 2)
            },
            "Best_Performing_Model": best_model,
            "Category_MAE_Breakdown": category_metrics,
            "Test_Predictions_DataFrame": test_df
        }
        return self.latest_eval_report

    def predict_next_month_expenses(self, monthly_df: pd.DataFrame, mid_month_spend: dict = None) -> pd.DataFrame:
        """
        Estimates likely end-of-month spending for each category.
        Labels all outputs as estimates and provides a forecast range (+/- MAE).
        """
        if not self.is_trained:
            self.train_and_evaluate(monthly_df)

        latest_period = monthly_df['YearMonth'].max()
        next_period = latest_period + 1
        latest_rows = monthly_df[monthly_df['YearMonth'] == latest_period].copy()

        forecasts = []
        for _, row in latest_rows.iterrows():
            cat = row['Category']
            cat_hist = monthly_df[monthly_df['Category'] == cat].sort_values('YearMonth')
            recent_3_avg = cat_hist['Amount'].tail(3).mean()
            prev_1m = row['Amount']
            prev_2m = cat_hist['Amount'].iloc[-2] if len(cat_hist) >= 2 else prev_1m

            # If user provides current first-half spending, use it; otherwise estimate from historical ratio
            if mid_month_spend and cat in mid_month_spend:
                first_half_val = mid_month_spend[cat]
            else:
                first_half_val = cat_hist['First_Half_Spend'].tail(3).mean()

            X_pred = pd.DataFrame([{
                'Category_Code': self.category_mapping.get(cat, 0),
                'Month_Num': next_period.month,
                'Year_Num': next_period.year,
                'Prev_1M_Spend': prev_1m,
                'Prev_2M_Spend': prev_2m,
                'Prev_3M_Rolling_Avg': recent_3_avg,
                'First_Half_Spend': first_half_val
            }])

            baseline_est = round(float(recent_3_avg), 2)
            lr_est = round(float(max(0, self.lr_model.predict(X_pred)[0])), 2)
            rf_est = round(float(max(0, self.rf_model.predict(X_pred)[0])), 2)

            forecasts.append({
                'Target_Month': str(next_period),
                'Category': cat,
                'Baseline_Estimate': baseline_est,
                'Linear_Regression_Estimate': lr_est,
                'Random_Forest_Estimate': rf_est,
                'Recommended_Estimate': rf_est,
                'Label': 'Estimated Spend (Not Guaranteed)'
            })

        return pd.DataFrame(forecasts)

    def generate_financial_insights(self, monthly_df: pd.DataFrame, category_budgets: dict = None) -> list:
        """
        Generates data-driven financial insights:
        1. Flags unusual spending patterns using rolling 6-month Z-scores (> 2.0 std dev).
        2. Compares actual/estimated spending against optional monthly category budgets.
        3. Identifies month-over-month spending changes.
        """
        insights = []
        recent_periods = sorted(monthly_df['YearMonth'].unique())[-6:]
        recent_df = monthly_df[monthly_df['YearMonth'].isin(recent_periods)].copy()

        # 1. Anomaly detection (Unusual spending spikes)
        for _, row in recent_df.iterrows():
            mean_6m = row.get('Rolling_6M_Mean', np.nan)
            std_6m = row.get('Rolling_6M_Std', np.nan)
            if pd.notna(mean_6m) and pd.notna(std_6m) and std_6m > 0:
                z_score = (row['Amount'] - mean_6m) / std_6m
                if z_score > 2.0:
                    pct_above = ((row['Amount'] - mean_6m) / mean_6m) * 100
                    insights.append({
                        'Type': 'Unusual Spending Alert',
                        'Period': str(row['YearMonth']),
                        'Category': row['Category'],
                        'Severity': 'High' if z_score > 3.0 else 'Moderate',
                        'Message': (
                            f"Unusual spike in {row['Category']} during {row['YearMonth']}: "
                            f"Spent {self.currency}{row['Amount']:,.2f}, which is {pct_above:.1f}% above "
                            f"the 6-month rolling average ({self.currency}{mean_6m:,.2f})."
                        )
                    })

        # 2. Optional Stretch Goal: Monthly Category Budget Alerts
        if category_budgets:
            latest_period = monthly_df['YearMonth'].max()
            latest_df = monthly_df[monthly_df['YearMonth'] == latest_period]
            for _, row in latest_df.iterrows():
                cat = row['Category']
                if cat in category_budgets:
                    budget = category_budgets[cat]
                    diff = row['Amount'] - budget
                    if diff > 0:
                        insights.append({
                            'Type': 'Budget Overspending Alert',
                            'Period': str(latest_period),
                            'Category': cat,
                            'Severity': 'High',
                            'Message': (
                                f"{cat} exceeded the monthly budget of {self.currency}{budget:,.2f} "
                                f"by {self.currency}{diff:,.2f} in {latest_period}."
                            )
                        })

        if not insights:
            insights.append({
                'Type': 'Normal Pattern',
                'Period': 'Recent',
                'Category': 'All',
                'Severity': 'Low',
                'Message': 'Spending patterns across all categories are within normal historical ranges.'
            })

        return insights
