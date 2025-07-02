# Marketing Analytics Streamlit Dashboard

This repository contains a Streamlit application for **visualising and modelling** the social‑media engagement dataset (`data/social_media_usage.csv`).

## Tabs & Features

1. **Data Visualisation** – 10+ interactive charts with filters  
2. **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting  
3. **Clustering** – K‑means with elbow plot & persona table  
4. **Association Rules** – Apriori mining with adjustable thresholds  
5. **Regression Insights** – Linear, Ridge, Lasso, Decision‑Tree Regressor

The app loads the bundled dataset by default, but you can still **upload your own file** in the sidebar to override it.

## Local dev

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to GitHub  
2. In Streamlit Cloud: *New app* → pick repo → select `app.py`  
3. Click **Deploy**

Enjoy!
