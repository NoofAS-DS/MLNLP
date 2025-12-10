# =========================================
# Job Level Prediction Dashboard
# (Tabular Model + NLP Model + Scaler)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Job Level Prediction",
    layout="wide"
)

st.title("🎯 Job Level Prediction Dashboard")
st.caption(
    "التنبؤ بمستوى الوظيفة باستخدام البيانات الهيكلية (Tabular) "
    "+ الوصف الوظيفي (NLP)"
)

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    raw_df = pd.read_csv("data_jobs.csv")                 # البيانات الأصلية
    prep_df = pd.read_csv("smaller_df_prepared.csv")      # بعد الـ Encoding
    return raw_df, prep_df


# ---------------------------------
# Load Models
# ---------------------------------
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_jobs_level_model.pkl")
    scaler = joblib.load("scaler_jobs_level.pkl")         # ✅ scaler
    tfidf = joblib.load("tfidf_description.pkl")
    nlp_model = joblib.load("nlp_level_model.pkl")
    return rf_model, scaler, tfidf, nlp_model


raw_df, prep_df = load_data()
rf_model, scaler, tfidf, nlp_model = load_models()

# ---------------------------------
# Prepare X / y
# ---------------------------------
X_tabular = prep_df.drop("level", axis=1)
y_true = prep_df["level"]

# ---------------------------------
# Label Mapping
# ---------------------------------
level_map = {
    0: "Not Applicable",
    1: "Internship",
    2: "Entry level",
    3: "Associate",
    4: "MidSenior level",
    5: "Director",
    6: "Executive"
}

level_labels = [level_map[i] for i in range(7)]

# ---------------------------------
# Sidebar - Job Selection
# ---------------------------------
st.sidebar.header("🔎 اختيار وظيفة")

def job_label(row):
    return f"{row.name} | {str(row['position'])[:35]} | {row['city']}"

options = raw_df.apply(job_label, axis=1).tolist()
choice = st.sidebar.selectbox("اختر وظيفة:", options)

selected_index = int(choice.split("|")[0].strip())

alpha = st.sidebar.slider(
    "وزن المودل الجدولي (RF) في الدمج",
    0.0, 1.0, 0.5, 0.05
)

# ---------------------------------
# Selected Row
# ---------------------------------
row_raw = raw_df.loc[selected_index]
row_tabular = X_tabular.loc[[selected_index]]

true_level = level_map[y_true.loc[selected_index]]

# ✅ تجهيز البيانات (Scaler موجود للاستخدام لو احتجناه)
row_tabular_scaled = scaler.transform(row_tabular)

# ---------------------------------
# Tabular Prediction (Random Forest)
# ---------------------------------
proba_rf = rf_model.predict_proba(row_tabular)[0]
pred_rf = level_map[int(np.argmax(proba_rf))]

# ---------------------------------
# NLP Prediction
# ---------------------------------
desc = str(row_raw["description"])
X_desc = tfidf.transform([desc])

proba_nlp = nlp_model.predict_proba(X_desc)[0]
pred_nlp = level_map[int(np.argmax(proba_nlp))]

# ---------------------------------
# Ensemble Prediction
# ---------------------------------
beta = 1 - alpha
proba_ensemble = alpha * proba_rf + beta * proba_nlp
pred_ensemble = level_map[int(np.argmax(proba_ensemble))]

# ---------------------------------
# Job Information
# ---------------------------------
st.subheader("📄 معلومات الوظيفة")

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**المسمى الوظيفي:** {row_raw['position']}")
    st.markdown(f"**الشركة:** {row_raw['company']}")
    st.markdown(f"**المدينة:** {row_raw['city']}")
    st.markdown(f"**المؤهل:** {row_raw['degrees']}")

with c2:
    st.markdown(f"**Total Positions:** {row_raw['TotalPositions']}")
    st.markdown(f"**Positions / Month:** {row_raw['PositionsByMonth']}")
    st.markdown(f"**Years of Experience:** {row_raw['year_of_ex']}")

st.markdown("### 📝 الوصف الوظيفي")
st.write(desc)

# ---------------------------------
# Predictions
# ---------------------------------
st.subheader("🔮 نتائج التنبؤ")

p1, p2, p3, p4 = st.columns(4)

p1.metric("✅ المستوى الحقيقي", true_level)
p2.metric("🌳 Random Forest", pred_rf)
p3.metric("🧠 NLP Model", pred_nlp)
p4.metric("⚖️ Ensemble", pred_ensemble)

# ---------------------------------
# Probability Charts
# ---------------------------------
st.subheader("📊 احتمالات كل فئة")

prob_df = pd.DataFrame({
    "Level": level_labels,
    "RandomForest": proba_rf,
    "NLP": proba_nlp,
    "Ensemble": proba_ensemble
})

tab1, tab2, tab3 = st.tabs(["🌳 RF", "🧠 NLP", "⚖️ Ensemble"])

with tab1:
    fig = px.bar(prob_df, x="Level", y="RandomForest",
                 title="Random Forest Probabilities")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.bar(prob_df, x="Level", y="NLP",
                 title="NLP Model Probabilities")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.bar(prob_df, x="Level", y="Ensemble",
                 title="Ensemble Probabilities")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Educational Notes
# ---------------------------------
with st.expander("📚 ملاحظات تعليمية"):
    st.markdown("""
- **Random Forest** لا يحتاج Scaling  
- **Logistic Regression / KNN / SVM** تحتاج Scaler  
- لذلك قمنا بتحميل **scaler_jobs_level.pkl** لضمان:
  - نفس المعالجة المستخدمة أثناء التدريب
  - جاهزية النظام لتغيير المودل مستقبلًا
- مودل **NLP** يعتمد فقط على الوصف النصي
- **Ensemble** يدمج المودلين لتحسين العدالة بين الفئات
""")
