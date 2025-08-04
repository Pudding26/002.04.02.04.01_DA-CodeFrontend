import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RegressionPlot:
    def __init__(self, df, y_col, X_cols,
                 model_type="linear", degree=2,
                 train_size=0.8, quick_mode=False,
                 early_stopping_rounds=10):   # 👈 default here
        self.df = df
        self.y_col = y_col
        self.X_cols = X_cols
        self.model_type = model_type
        self.degree = degree
        self.train_size = train_size
        self.quick_mode = quick_mode
        self.early_stopping_rounds = early_stopping_rounds   # 👈 always defined


    def run(self):
        X = self.df[self.X_cols].dropna()
        y = self.df.loc[X.index, self.y_col]

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=42, shuffle=True
        )

        # ----- Model selection -----
        if self.model_type == "linear":
            model = make_pipeline(StandardScaler(), LinearRegression())
            use_statsmodels = True
        elif self.model_type == "poly":
            model = make_pipeline(StandardScaler(),
                                  PolynomialFeatures(self.degree, include_bias=False),
                                  LinearRegression())
            use_statsmodels = False
        elif self.model_type == "rf":
            model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            use_statsmodels = False
        elif self.model_type == "xgboost":
            if self.quick_mode:
                # ⚡ quick mode
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="rmse"
                )
            else:
                # full mode
                model = XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="rmse"
                )
            use_statsmodels = False
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # ----- Fit -----
        if self.model_type == "xgboost":
            fit_kwargs = {
                "X": X_train,
                "y": y_train,
                "eval_set": [(X_test, y_test)],
                "verbose": False,
            }
            if self.quick_mode and self.early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

            model.fit(**fit_kwargs)
        else:
            model.fit(X_train, y_train)


        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "train": {
                "R²": r2_score(y_train, y_train_pred),
                "MAE": mean_absolute_error(y_train, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            },
            "test": {
                "R²": r2_score(y_test, y_test_pred),
                "MAE": mean_absolute_error(y_test, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            }
        }

        # Coefs / feature importances
        coef_df, pvalues_df = None, None
        if self.model_type == "linear":
            X_const = sm.add_constant(X_train)
            ols = sm.OLS(y_train, X_const).fit()
            coef_df = pd.DataFrame({
                "Feature": ["Intercept"] + self.X_cols,
                "Coefficient": ols.params.values,
                "p-value": ols.pvalues.values
            })
            pvalues_df = coef_df
        elif self.model_type in ["rf", "xgboost"]:
            coef_df = pd.DataFrame({
                "Feature": self.X_cols,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

        # --- Figures ---
        figs = {}

        if coef_df is not None:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            plot_y = "Coefficient" if "Coefficient" in coef_df.columns else "Importance"
            sns.barplot(data=coef_df[coef_df["Feature"] != "Intercept"]
                        if "Intercept" in coef_df["Feature"].values else coef_df,
                        x="Feature", y=plot_y, ax=ax1)
            ax1.tick_params(axis="x", rotation=45)
            figs["coef_barplot"] = fig1
            figs["coef_table"] = coef_df

        # Predicted vs Actual (test set)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.scatter(y_test, y_test_pred, alpha=0.5)
        ax2.set_xlabel("Actual (Test)")
        ax2.set_ylabel("Predicted (Test)")
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        figs["pred_vs_actual"] = fig2

        # Residuals (test set)
        residuals = y_test - y_test_pred
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, bins=50, kde=True, ax=ax3)
        ax3.set_title("Residuals (Test)")
        figs["residuals"] = fig3

        # Correlation heatmap
        corr = self.df[self.X_cols + [self.y_col]].corr()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        ax4.set_title("Correlation Heatmap")
        figs["correlation"] = fig4

        return {
            "metrics": metrics,
            "coef_table": coef_df,
            "pvalues": pvalues_df,
            "figs": figs,
        }
