##############################################################################
# app.py – Marketing / Social-Media Analytics Dashboard
##############################################################################
import base64
import io
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, recall_score,
                             r2_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 1.  Config & constants
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Marketing Analytics", layout="wide")

REPO_ROOT = Path(__file__).parent
DATA_PATH  = REPO_ROOT / "data" / "social_media_usage.csv"   # optional starter
MODEL_DIR  = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# 2.  Utility helpers
# --------------------------------------------------------------------------- #
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load the baked-in sample dataset if it exists; otherwise return empty df."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    st.info("🔄 No default dataset found. Please upload a CSV/Excel file to begin.")
    return pd.DataFrame()

def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    return load_default_data()

def build_preprocessor(df: pd.DataFrame, target: str):
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]
    return ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

def metrics_clf(y, pred):
    return dict(
        Accuracy  = accuracy_score(y, pred),
        Precision = precision_score(y, pred, zero_division=0),
        Recall    = recall_score(y, pred, zero_division=0),
        F1        = f1_score(y, pred, zero_division=0),
    )

def metrics_reg(y, pred):
    return dict(
        R2   = r2_score(y, pred),
        RMSE = mean_squared_error(y, pred, squared=False),
    )

def download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'

# --------------------------------------------------------------------------- #
# 3.  Sidebar – data ingestion
# --------------------------------------------------------------------------- #
st.sidebar.header("Upload or use sample data")
uploaded_file = st.sidebar.file_uploader("CSV or Excel", ["csv", "xlsx", "xls"])
df = load_data(uploaded_file)

if df.empty:
    st.stop()         # short-circuit if still no data after prompt

st.sidebar.subheader("Preview")
st.sidebar.write(df.head())
st.sidebar.write(f"Rows {df.shape[0]} × Cols {df.shape[1]}")

# --------------------------------------------------------------------------- #
# 4.  Tabs
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualisation", "Classification", "Clustering",
     "Association Rules", "Regression"]
)

##############################################################################
# 4-A  DATA VISUALISATION
##############################################################################
with tab1:
    st.header("🔍 Exploratory Insights")

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- simple filters ---------------------------------------------------- #
    with st.expander("Filters"):
        filtered = df.copy()
        for c in cat_cols:
            opts = st.multiselect(f"{c} values", filtered[c].unique(),
                                  default=filtered[c].unique())
            filtered = filtered[filtered[c].isin(opts)]

    # --- distributions ----------------------------------------------------- #
    st.subheader("Distributions & Relationships")
    for col in num_cols[:3]:
        st.plotly_chart(
            px.histogram(filtered, x=col, nbins=30,
                         title=f"Distribution of {col}"),
            use_container_width=True,
        )
    # correlation
    if len(num_cols) >= 2:
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=True)

        st.plotly_chart(
            px.scatter_matrix(filtered[num_cols], title="Scatter-Matrix"),
            use_container_width=True,
        )

    # category vs numeric
    if cat_cols and num_cols:
        for c in cat_cols[:2]:
            st.plotly_chart(
                px.bar(filtered, x=c, y=num_cols[0],
                       title=f"{num_cols[0]} by {c}", barmode="group"),
                use_container_width=True
            )

##############################################################################
# 4-B  CLASSIFICATION
##############################################################################
with tab2:
    st.header("🎯 Binary Classification")
    tgt_cols = df.select_dtypes(include=["int64", "int32", "bool"]).columns.tolist()
    target = st.selectbox("Choose target (binary)", tgt_cols)
    if target:
        X = df.drop(columns=[target])
        y = df[target].astype(int)
        pre = build_preprocessor(df, target)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)
        models = {
            "KNN"             : KNeighborsClassifier(),
            "DecisionTree"    : DecisionTreeClassifier(random_state=42),
            "RandomForest"    : RandomForestClassifier(n_estimators=200,
                                                       random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }

        perf, probs = {}, {}
        for name, mdl in models.items():
            pipe = Pipeline([("pre", pre), ("mdl", mdl)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            perf[name] = metrics_clf(y_te, pred)
            probs[name] = pipe.predict_proba(X_te)[:, 1]
            joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")

        st.subheader("Performance table")
        st.dataframe(pd.DataFrame(perf).T.style.format("{:.2%}"),
                     use_container_width=True)

        # Confusion-matrix toggle
        model_sel = st.selectbox("Confusion matrix for model:", list(models.keys()))
