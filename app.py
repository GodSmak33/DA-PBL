import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import joblib, base64, io, warnings
warnings.filterwarnings('ignore')

##############################################################################
# Helper functions
##############################################################################

DATA_PATH = "data/social_media_usage.csv"

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

def load_data(upload):
    if upload is not None:
        if upload.name.endswith(".csv"):
            return pd.read_csv(upload)
        else:
            return pd.read_excel(upload)
    return load_default_data()

def build_preprocessor(df, target):
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

def metrics_classification(y_true, y_pred):
    return dict(
        Accuracy = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall = recall_score(y_true, y_pred, zero_division=0),
        F1 = f1_score(y_true, y_pred, zero_division=0)
    )

def metrics_regression(y_true, y_pred):
    return dict(
        R2 = r2_score(y_true, y_pred),
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
    )

def download_link(df, filename, label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

##############################################################################
# Streamlit layout
##############################################################################
st.set_page_config(page_title="Marketing Analytics", layout="wide")
st.title("📊 Marketing Analytics Dashboard")

# Sidebar – data upload
st.sidebar.header("Upload / Replace Dataset")
upload = st.sidebar.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
df = load_data(upload)

# Quick info
st.sidebar.subheader("Dataset snapshot")
st.sidebar.write(df.head())
st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Visualisation","Classification","Clustering","Association Rules","Regression"]
)

##############################################################################
# 1. Data Visualisation
##############################################################################
with tab1:
    st.header("Exploratory Data Analysis")
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Filter section
    with st.expander("Filters", expanded=False):
        choose_cat = st.multiselect("Categorical columns to filter", cat_cols)
        filtered = df.copy()
        for col in choose_cat:
            vals = st.multiselect(f"Values for {col}", filtered[col].unique(), default=filtered[col].unique())
            filtered = filtered[filtered[col].isin(vals)]

    # Auto‑charts – at least 10 insights
    st.subheader("Distributions")
    for n in num_cols[:3]:
        fig = px.histogram(filtered, x=n, nbins=30, title=f"Distribution of {n}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlations")
    corr = filtered[num_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if len(num_cols) >= 2:
        fig = px.scatter_matrix(filtered[num_cols])
        st.plotly_chart(fig, use_container_width=True)

    if cat_cols and num_cols:
        for c in cat_cols[:2]:
            fig = px.bar(filtered, x=c, y=num_cols[0], title=f"{num_cols[0]} by {c}")
            st.plotly_chart(fig, use_container_width=True)

##############################################################################
# 2. Classification
##############################################################################
with tab2:
    st.header("Binary Classification")
    potential_targets = df.select_dtypes(include=['int64','int32','bool']).columns.tolist()
    target = st.selectbox("Choose target", potential_targets, index=len(potential_targets)-2)
    if target:
        X = df.drop(columns=[target])
        y = df[target].astype(int)
        pre = build_preprocessor(df, target)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

        models = {
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }

        perf, probas = {}, {}
        for name, mdl in models.items():
            pipe = Pipeline([("pre", pre), ("mdl", mdl)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            perf[name] = metrics_classification(y_test, pred)
            probas[name] = pipe.predict_proba(X_test)[:,1]
            joblib.dump(pipe, f"{name}_clf.pkl")

        st.subheader("Performance table")
        st.dataframe(pd.DataFrame(perf).T.style.format("{:.2%}"))

        # Confusion matrix
        choose = st.selectbox("Confusion matrix for:", list(models.keys()))
        cm = confusion_matrix(y_test, Pipeline([("pre",pre),("mdl",models[choose])]).fit(X_train,y_train).predict(X_test))
        fig = px.imshow(cm, text_auto=True,
                        x=["Pred 0","Pred 1"], y=["Actual 0","Actual 1"],
                        title=f"Confusion Matrix – {choose}")
        st.plotly_chart(fig, use_container_width=True)

        # ROC
        st.subheader("ROC curve")
        fig = go.Figure()
        for name, p in probas.items():
            fpr, tpr, _ = roc_curve(y_test, p)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)

        # Predict new
        st.subheader("Predict on new data")
        new_file = st.file_uploader("Upload new rows (no target)", type=["csv","xlsx","xls"])
        if new_file:
            new_df = load_data(new_file)
            mdl_name = st.selectbox("Model", list(models.keys()))
            mdl = joblib.load(f"{mdl_name}_clf.pkl")
            preds = mdl.predict(new_df)
            new_df["prediction"] = preds
            st.write(new_df.head())
            st.markdown(download_link(new_df,"predictions.csv","Download predictions"), unsafe_allow_html=True)

##############################################################################
# 3. Clustering
##############################################################################
with tab3:
    st.header("K‑means Clustering")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feats = st.multiselect("Numeric features", num_cols, default=num_cols)
    if len(feats) >= 2:
        X = df[feats].dropna()
        X_scaled = StandardScaler().fit_transform(X)

        st.subheader("Elbow plot")
        inertias = []
        k_range = range(2,11)
        for k in k_range:
            inertias.append(KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled).inertia_)
        fig = go.Figure(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers"))
        fig.update_layout(xaxis_title="k", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)

        k = st.slider("Choose k", 2,10,3)
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X_scaled)
        df_cluster = df.copy()
        df_cluster["cluster"] = labels
        st.subheader("Cluster persona")
        st.dataframe(df_cluster.groupby("cluster")[feats].mean().round(2))

        st.markdown(download_link(df_cluster,"clustered.csv","Download dataset with clusters"), unsafe_allow_html=True)

##############################################################################
# 4. Association Rules
##############################################################################
with tab4:
    st.header("Association rule mining")
    cats = df.select_dtypes(include=['object','category']).columns.tolist()
    cols = st.multiselect("Choose categorical columns", cats, default=cats[:2])
    if len(cols) >= 2:
        basket = pd.get_dummies(df[cols].astype(str))
        support = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        freq = apriori(basket, min_support=support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.1)
        st.dataframe(rules.sort_values("lift", ascending=False).head(10))

##############################################################################
# 5. Regression Insights
##############################################################################
with tab5:
    st.header("Quick regression benchmarks")
    numeric = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique()>10]
    target = st.selectbox("Target (numeric)", numeric)
    if target:
        X = df.drop(columns=[target]).select_dtypes(include=np.number)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        regs = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
        }
        results = {}
        for n, r in regs.items():
            r.fit(X_train, y_train)
            results[n] = metrics_regression(y_test, r.predict(X_test))
        st.dataframe(pd.DataFrame(results).T)
