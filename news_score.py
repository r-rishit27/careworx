import os
from pathlib import Path
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder



cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)


icu_csv_path = "icu.csv"


if not os.path.exists(icu_csv_path):
    raise FileNotFoundError("The ICU dataset is missing. Please upload it.")


df = pd.read_csv(icu_csv_path, usecols=["GatewayName", "HR", "NIBP_Systolic", "NIBP_Diastolic", "SpO2", "RR",])


# Define the NEWS scoring function
def calculate_news(hr, systolic_bp, diastolic_bp, spo2, rr):
    score = 0
    
    # Respiration Rate (RR)
    if rr <= 8 or rr >= 25:
        score += 3
    elif 9 <= rr <= 11 or 21 <= rr <= 24:
        score += 2
    elif 12 <= rr <= 20:
        score += 0
    
    # Oxygen Saturation (SpO2)
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1
    elif spo2 >= 96:
        score += 0
    
    # Systolic Blood Pressure
    if systolic_bp <= 90 or systolic_bp >= 220:
        score += 3
    elif 91 <= systolic_bp <= 100:
        score += 2
    elif 101 <= systolic_bp <= 110:
        score += 1
    elif 111 <= systolic_bp <= 219:
        score += 0
    
    # Heart Rate (HR)
    if hr <= 40 or hr >= 131:
        score += 3
    elif 41 <= hr <= 50 or 111 <= hr <= 130:
        score += 2
    elif 51 <= hr <= 90:
        score += 0
    elif 91 <= hr <= 110:
        score += 1
    
    return score

# Calculate NEWS score for each timestamp
df["NEWS_Score"] = df.apply(lambda row: calculate_news(row["HR"], row["NIBP_Systolic"], row["NIBP_Diastolic"], row["SpO2"], row["RR"]), axis=1)

# Aggregate by patient and take the average of recorded scores
avg_news_df = df.groupby("GatewayName")["NEWS_Score"].mean().reset_index()

def main():
    st.set_page_config(page_title="NEWS Agent", layout="wide")
    st.title("🚑 NEWS Agent - National Early Warning Score Calculator")
    st.write("Enter a patient ID to compute the NEWS score or view high-risk patients.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        patient_id = st.text_input("🔍 Enter Patient ID (GatewayName)")
    
    if st.button("📊 Calculate NEWS Score"):
        patient_data = avg_news_df[avg_news_df["GatewayName"] == patient_id]
        
        if patient_data.empty:
            st.error("❌ Patient ID not found.")
        else:
            score = patient_data["NEWS_Score"].values[0]
            st.success(f"✅ The average NEWS score for {patient_id} is: {score:.2f}")
            
            if score >6:
                st.error("🚨 High risk! Immediate medical intervention required.")
            elif 4< score <=6:
                st.warning("⚠️ Medium risk. Consider closer observation.")
            elif 1 <= score <= 4:
                st.info("ℹ️ Low risk, monitor periodically.")
            else:
                st.success("🟢 No immediate risk detected.")
    
    with col2:
        st.subheader("🚨 High-Risk Patients")
        high_risk_df = avg_news_df[avg_news_df["NEWS_Score"] >6][["GatewayName", "NEWS_Score"]]
        
        if not high_risk_df.empty:
            gb = GridOptionsBuilder.from_dataframe(high_risk_df)
            gb.configure_pagination(enabled=True)
            gb.configure_side_bar()
            grid_options = gb.build()
            AgGrid(high_risk_df, gridOptions=grid_options, theme="streamlit")
        else:
            st.success("No high-risk patients detected.")

if __name__ == "__main__":
    main()
