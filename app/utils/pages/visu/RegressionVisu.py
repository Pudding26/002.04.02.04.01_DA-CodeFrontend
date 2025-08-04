import streamlit as st
from app.utils.pages.visu.regression.RegressionPlot import RegressionPlot


class RegressionVisu:
    def __init__(self, df, shared_state=None):
        self.df = df
        self.shared_state = shared_state or {}

    def render(self):
        st.header("Regression Analysis")

        numeric_cols = list(self.df.select_dtypes(include="number").columns)

        # --- User selections ---
        y_col = st.selectbox("Target (Y)", numeric_cols)
        X_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != y_col])

        model_type = st.selectbox("Model type", ["linear", "poly", "rf", "xgboost"])
        degree = None
        if model_type == "poly":
            degree = st.slider("Polynomial degree", 2, 5, 2)

        quick_mode = False
        if model_type == "xgboost":
            quick_mode = st.checkbox("⚡ Quick Mode (faster, less accurate)", value=True)

        train_size = st.slider("Train size (%)", 50, 95, 80) / 100.0

        if not y_col or not X_cols:
            st.warning("Select at least one target and one feature.")
            return

        # --- Run regression ---
        plotter = RegressionPlot(self.df, y_col=y_col, X_cols=X_cols,
                                 model_type=model_type, degree=degree,
                                 train_size=train_size, quick_mode=quick_mode)
        results = plotter.run()

        # --- Metrics ---
        st.subheader("Model Performance")
        st.write("**Train Metrics:**", results["metrics"]["train"])
        st.write("**Test Metrics:**", results["metrics"]["test"])

        # --- Coefficients or Feature Importances ---
        if results["coef_table"] is not None:
            label = "Model Coefficients" if model_type in ["linear", "poly"] else "Feature Importances"
            st.subheader(label)
            st.dataframe(results["coef_table"])
            st.pyplot(results["figs"]["coef_barplot"])

        # --- P-values (only for OLS) ---
        if results["pvalues"] is not None:
            st.subheader("P-values (OLS)")
            st.dataframe(results["pvalues"])

        # --- Pred vs Actual ---
        st.subheader("Predicted vs Actual (Test)")
        st.pyplot(results["figs"]["pred_vs_actual"])

        # --- Residuals ---
        st.subheader("Residuals Distribution (Test)")
        st.pyplot(results["figs"]["residuals"])

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap")
        st.pyplot(results["figs"]["correlation"])
