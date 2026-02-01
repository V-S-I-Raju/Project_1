import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Advisor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("student_performance_model.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>
        🎓 Student Performance Improvement Advisor
    </h1>
    <p style='text-align: center; font-size: 16px; color: #555;'>
        ML-based prediction with intelligent habit analysis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Inputs
# -----------------------------
st.subheader("📥 Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📘 Study Hours / Day", 0, 12, 4)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)

with col2:
    sleep_hours = st.slider("😴 Sleep Hours / Day", 4, 10, 7)
    mobile_hours = st.slider("📱 Mobile Usage (Hours / Day)", 0, 10, 3)

# -----------------------------
# Prediction
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Performance", use_container_width=True):

    # ML Prediction
    input_data = np.array([[study_hours, attendance, sleep_hours, mobile_hours]])
    predicted_score = model.predict(input_data)[0]

    # -----------------------------
    # Intelligent Sleep Adjustment
    # -----------------------------
    penalty_reason = None

    if sleep_hours > 9:
        predicted_score -= 5
        penalty_reason = "Oversleeping (>9 hours) may reduce alertness."

    predicted_score = round(predicted_score, 2)

    # -----------------------------
    # Results
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Exam Score", predicted_score)

    # Performance Level
    if predicted_score >= 80:
        st.success("🌟 Performance Level: Excellent")
    elif predicted_score >= 60:
        st.info("👍 Performance Level: Average")
    else:
        st.warning("⚠ Performance Level: Needs Improvement")

    # -----------------------------
    # Advice
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📝 Personalized Improvement Advice")

    advice_given = False

    if study_hours < 4:
        st.write("📘 Increase study time to at least **4–6 hours per day**.")
        advice_given = True

    if attendance < 75:
        st.write("🏫 Improve attendance for better academic consistency.")
        advice_given = True

    if sleep_hours < 6:
        st.write("😴 Sleeping less than 6 hours affects memory and focus.")
        advice_given = True

    if sleep_hours > 9:
        st.write("⏰ Too much sleep can reduce daily discipline and energy.")
        advice_given = True
