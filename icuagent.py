import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.ollama import Ollama
from st_aggrid import AgGrid, GridOptionsBuilder

load_dotenv()
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
tmp.mkdir(exist_ok=True, parents=True)

icu_csv_path = "icu.csv"

if not os.path.exists(icu_csv_path):
    raise FileNotFoundError(f"The file {icu_csv_path} does not exist. Please upload it.")


df = pd.read_csv(icu_csv_path, usecols=["GatewayName", "HR", "NIBP_Systolic", "NIBP_Diastolic", "SpO2", "RR", "Timestamp"])

#news score calculation 

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

# Bradycardia and Tachycardia Detection
df.sort_values(by=["GatewayName", "Timestamp"], inplace=True)
df["HR_Change"] = df.groupby("GatewayName")["HR"].diff()
df["Condition"] = "Normal"
df.loc[df["HR_Change"] > 25, "Condition"] = "Tachycardia"
df.loc[df["HR_Change"] < -15, "Condition"] = "Bradycardia"
critical_patients = df[df["Condition"] != "Normal"]

#Agent Framework
csv_agent = PythonAgent(
    model=Ollama(id="llama3.3"),
    base_dir=tmp,
    files=[CsvFile(path=icu_csv_path, description="ICU patient vitals monitoring data, including heart rate, oxygen levels, blood pressure, respiration rate, and other critical parameters.")],
    markdown=True,
    pip_install=True,
    show_tool_calls=True,
    system_prompt=(
        "You are an AI-powered ICU monitoring assistant specializing in early warning system (EWS) detection, "
        "including NEWS (National Early Warning Score) assessment. Your task is to analyze real-time ICU patient vitals, "
        "identify critical conditions such as bradycardia, tachycardia, hypoxia, and sepsis risk, and provide actionable alerts "
        "to doctors and medical staff. \n\n"
        "Key responsibilities:\n"
        "- Monitor heart rate, blood pressure, oxygen saturation (SpO2), respiration rate, and temperature.\n"
        "- Detect anomalies using predefined medical thresholds (e.g., NEWS score calculation, identifying bradycardia/tachycardia events).\n"
        "- Provide real-time alerts when a patient's condition becomes critical.\n"
        "- Suggest possible medical interventions based on the detected anomalies.\n"
        "- Ensure alerts are clear, concise, and medically relevant to assist in quick decision-making.\n\n"
        "You must prioritize patient safety, minimize false alarms, and escalate alerts appropriately when needed."
    )
)

# Streamlit UI
def main():
    st.set_page_config(page_title="ICU Monitoring System", layout="wide")
    st.title("🚑 ICU Patient Monitoring & Alert System")

    # NEWS Score Table
    st.subheader("📊 NEWS Score Analysis")
    news_table = df[["GatewayName", "Timestamp", "NEWS_Score", "Warning_Message"]]
    gb = GridOptionsBuilder.from_dataframe(news_table)
    gb.configure_pagination()
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(news_table, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)
    
    st.subheader("📊 NEWS Score ")
    # Gateway filter
    gateways = df["GatewayName"].unique()
    selected_gateway = st.selectbox("Select Gateway", gateways)
    filtered_df = df[df["GatewayName"] == selected_gateway]
    
    # Plot Spike Detection
    parameter = st.selectbox("Select Parameter for Spike Detection", ["NEWS_Score", "HR", "NIBP_Systolic", "RR"])
    
    if parameter == "NEWS_Score":
        filtered_df = filtered_df[filtered_df['NEWS_Score'] > 5]
        title = "NEWS Score Spikes"
    elif parameter == "HR":
        filtered_df = filtered_df[filtered_df['HR']>140]
        title = "Heart Rate Sudden Spikes"
    elif parameter == "NIBP_Systolic":
        filtered_df = filtered_df[filtered_df['NIBP_Systolic']>180]
        title = "Blood Pressure Sudden Spikes"
    elif parameter == "RR":
        filtered_df =filtered_df[filtered_df['RR']>30]
        title = "Respiration Rate Sudden Spikes"
    
    st.subheader(f"📈 {title}")
    fig = px.line(filtered_df, x="Timestamp", y=parameter, color="GatewayName", markers=True)
    fig.update_traces(mode="lines+markers", marker=dict(size=6, color="red"))
    st.plotly_chart(fig, use_container_width=True)

    # Bradycardia & Tachycardia Table
    st.subheader("⚠️ Critical Patients (Bradycardia & Tachycardia)")
    if not critical_patients.empty:
        critical_table = critical_patients[["GatewayName", "Timestamp", "HR", "HR_Change", "Condition"]]
        gb = GridOptionsBuilder.from_dataframe(critical_table)
        gb.configure_pagination()
        grid_options = gb.build()
        AgGrid(critical_table, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)
    else:
        st.success("No critical patients detected.")

    # User query input for AI agent
    st.subheader("🧠 AI-Powered ICU Data Search")
    query = st.text_area("Enter your question about the ICU data:", placeholder="e.g., How many patients have bradycardia?")
    if st.button("Search ICU Data"):
        if not query.strip():
            st.error("Please enter a valid query.")
            return
        try:
            with st.spinner("Processing your question..."):
                response = csv_agent.run(query)
                st.markdown(response.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
