# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

from prophet import Prophet
# TSFresh kept (optional usage later)
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="FitPulse – Health Analytics",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# USER DATABASE
# --------------------------------------------------
USER_DB = "users.csv"
if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB, index=False)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key in ["logged_in", "df", "metrics", "anomaly_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==================================================
# AUTHENTICATION
# ==================================================
def register():
    st.subheader("📝 Register")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    c = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        users = pd.read_csv(USER_DB)
        if u in users["username"].values:
            st.error("Username already exists")
        elif p != c:
            st.error("Passwords do not match")
        else:
            users.loc[len(users)] = [u, p]
            users.to_csv(USER_DB, index=False)
            st.success("Registered successfully")

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        users = pd.read_csv(USER_DB)
        if ((users.username == u) & (users.password == p)).any():
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    st.title("💓 FitPulse")
    choice = st.radio("Choose", ["Login", "Register"])
    login() if choice == "Login" else register()
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("💓 FitPulse")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Upload Data",
        "Analysis",
        "Anomalies",
        "TSFresh Features",   # ✅ NEW
        "Forecast",
        "Dashboard",
        "Reports",
        "About"
    ]
)


# ==================================================
# UPLOAD DATA
# ==================================================
if menu == "Upload Data":
    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "timestamp" not in df.columns:
            st.error("CSV must contain timestamp")
            st.stop()

        # 🔥 FIX 1: Convert timestamp and REMOVE timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp")

        metrics = df.select_dtypes(include=np.number).columns.tolist()

        st.session_state.df = df
        st.session_state.metrics = metrics

        st.success("✅ Dataset Loaded Successfully")
        st.dataframe(df.head())

# ==================================================
# ANALYSIS
# ==================================================
elif menu == "Analysis":
    st.title("📈 Exploratory Analysis")

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
    else:
        metric = st.selectbox("Metric", st.session_state.metrics)

        st.plotly_chart(
            px.line(df, x="timestamp", y=metric, markers=True),
            use_container_width=True
        )

        st.plotly_chart(
            px.histogram(df, x=metric, nbins=40),
            use_container_width=True
        )

        st.plotly_chart(
            px.box(df, y=metric),
            use_container_width=True
        )

        daily = df.resample("D", on="timestamp")[metric].mean().reset_index()
        st.plotly_chart(
            px.bar(daily, x="timestamp", y=metric),
            use_container_width=True
        )

# ==================================================
# ANOMALIES
# ==================================================
elif menu == "Anomalies":
    st.title("🚨 Anomaly Detection")

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
    else:
        metric = st.selectbox("Metric", st.session_state.metrics)
        method = st.selectbox(
            "Method",
            ["Statistical", "KMeans", "DBSCAN", "IsolationForest"]
        )

        X = df[[metric]].dropna()
        Xs = StandardScaler().fit_transform(X)

        if method == "Statistical":
            m, s = X.mean(), X.std()
            X["anomaly"] = abs(X - m) > 3 * s

        elif method == "KMeans":
            labels = KMeans(n_clusters=2, random_state=42).fit_predict(Xs)
            anomaly_cluster = pd.Series(labels).value_counts().idxmin()
            X["anomaly"] = labels == anomaly_cluster

        elif method == "DBSCAN":
            X["anomaly"] = DBSCAN(eps=0.8, min_samples=10).fit_predict(Xs) == -1

        else:
            X["anomaly"] = IsolationForest(
                contamination=0.05, random_state=42
            ).fit_predict(Xs) == -1

        anomalies = df.loc[X[X["anomaly"]].index]
        st.session_state.anomaly_df = anomalies

        fig = px.scatter(
            df,
            x="timestamp",
            y=metric,
            color=df.index.isin(anomalies.index),
            title="Anomaly Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)
        
# ==================================================
# TSFRESH FEATURES (SAFE MODE)
# ==================================================
elif menu == "TSFresh Features":
    st.title("🧠 TSFresh Feature Extraction (Safe Mode)")

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
        st.stop()

    metric = st.selectbox("Select Metric", st.session_state.metrics)

    if st.button("🚀 Generate TSFresh Features"):
        with st.spinner("Extracting TSFresh features..."):

            ts_df = df[["timestamp", metric]].copy()

            # 🔥 DROP NaNs (TSFresh requirement)
            ts_df = ts_df.dropna(subset=[metric])

            if ts_df.empty or len(ts_df) < 20:
                st.error("❌ Not enough clean data points for TSFresh")
                st.stop()

            ts_df["id"] = 1
            ts_df.rename(
                columns={"timestamp": "time", metric: "value"},
                inplace=True
            )

            # 🔥 Remove timezone
            ts_df["time"] = pd.to_datetime(ts_df["time"]).dt.tz_localize(None)

            try:
                features = extract_features(
                    ts_df,
                    column_id="id",
                    column_sort="time",
                    disable_progressbar=True
                )

                impute(features)

                st.success("✅ TSFresh Features Generated Successfully")
                st.write(f"🔢 Features extracted: {features.shape[1]}")
                st.dataframe(features.T.head(30))

                st.download_button(
                    "⬇ Download TSFresh Features",
                    features.to_csv(),
                    "tsfresh_features.csv"
                )

            except Exception as e:
                st.error("❌ TSFresh failed safely (app did not crash)")
                st.code(str(e))

        

# ==================================================
# FORECAST (PROPHET)
# ==================================================
elif menu == "Forecast":
    st.title("🔮 Forecasting")

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
    else:
        metric = st.selectbox("Metric", st.session_state.metrics)

        p_df = df[["timestamp", metric]].rename(
            columns={"timestamp": "ds", metric: "y"}
        )

        # 🔥 FIX 2: ENSURE Prophet-safe datetime
        p_df["ds"] = pd.to_datetime(p_df["ds"]).dt.tz_localize(None)

        model = Prophet()
        model.fit(p_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.plotly_chart(
            px.line(forecast, x="ds", y="yhat", title="30-Day Forecast"),
            use_container_width=True
        )

# ==================================================
# DASHBOARD
# ==================================================
elif menu == "Dashboard":
    st.title("📊 Smart Health Dashboard")

    df = st.session_state.df
    if df is not None:
        metric = st.selectbox("Metric", st.session_state.metrics)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg", round(df[metric].mean(), 2))
        c2.metric("Min", df[metric].min())
        c3.metric("Max", df[metric].max())
        c4.metric("Std", round(df[metric].std(), 2))

        st.plotly_chart(
            px.line(df, x="timestamp", y=metric),
            use_container_width=True
        )

# ==================================================
# REPORTS
# ==================================================
elif menu == "Reports":
    if st.session_state.df is not None:
        st.download_button(
            "Download CSV",
            st.session_state.df.to_csv(index=False),
            "fitpulse_report.csv"
        )

# ==================================================
# ABOUT
# ==================================================
elif menu == "About":
    st.markdown("""
    ## 💓 FitPulse – Advanced Health Analytics Platform

    **Features**
    - Secure Login & Registration
    - Automated Exploratory Data Analysis
    - Multi-Model Anomaly Detection
    - Time-Series Forecasting (Prophet)
    - Interactive & Animated Visualizations

    **Tech Stack**
    - Streamlit, Plotly
    - Scikit-learn
    - Prophet
    - TSFresh
    """)
