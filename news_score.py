import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

icu_csv_path = "icu.csv"

if not os.path.exists(icu_csv_path):
    raise FileNotFoundError("The ICU dataset is missing. Please upload it.")

df = pd.read_csv(icu_csv_path, usecols=["GatewayName", "HR", "NIBP_Systolic", "NIBP_Diastolic", "SpO2", "RR", "Timestamp"])

def calculate_news(hr, systolic_bp, diastolic_bp, spo2, rr):
    score = 0
    
    if rr <= 8 or rr >= 25:
        score += 3
    elif 9 <= rr <= 11 or 21 <= rr <= 24:
        score += 2
    elif 12 <= rr <= 20:
        score += 0
    
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1
    elif spo2 >= 96:
        score += 0
    
    if systolic_bp <= 90 or systolic_bp >= 220:
        score += 3
    elif 91 <= systolic_bp <= 100:
        score += 2
    elif 101 <= systolic_bp <= 110:
        score += 1
    elif 111 <= systolic_bp <= 219:
        score += 0
    
    if hr <= 40 or hr >= 131:
        score += 3
    elif 41 <= hr <= 50 or 111 <= hr <= 130:
        score += 2
    elif 51 <= hr <= 90:
        score += 0
    elif 91 <= hr <= 110:
        score += 1
    
    return score

df["NEWS_Score"] = df.apply(lambda row: calculate_news(row["HR"], row["NIBP_Systolic"], row["NIBP_Diastolic"], row["SpO2"], row["RR"]), axis=1)

def get_warning_message(score):
    if score > 6:
        return "🚨 High risk! Immediate medical intervention required."
    elif 4 < score <= 6:
        return "⚠️ Medium risk. Consider closer observation."
    elif 1 <= score <= 4:
        return "ℹ️ Low risk, monitor periodically."
    else:
        return "🟢 No immediate risk detected."

df["Warning_Message"] = df["NEWS_Score"].apply(get_warning_message)

high_risk_df = df[df["NEWS_Score"] > 4]

def main():
    st.set_page_config(page_title="NEWS Agent", layout="wide")
    st.title("🚑 NEWS Agent - National Early Warning Score Calculator")
    st.subheader(f"⚠️ Critical Patients Detected ")
    
    if not high_risk_df.empty:
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Gateway Name", "Timestamp", "NEWS Score", "Warning Message"],
                        fill_color='lightblue',
                        align='center',
                        font=dict(color='black', size=14)),
            cells=dict(values=[high_risk_df.GatewayName, 
                               high_risk_df.Timestamp,  
                               high_risk_df.NEWS_Score,
                               high_risk_df.Warning_Message],
                       fill_color='white',
                       align='center',
                       font=dict(color='black', size=12)))
        ])
        st.plotly_chart(fig)
    else:
        st.success("No high-risk patients detected.")

if __name__ == "__main__":
    main()
