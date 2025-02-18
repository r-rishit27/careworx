# Initialize AI-powered CSV agent
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
import speech_recognition as sr
import re


load_dotenv()


cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
tmp.mkdir(exist_ok=True, parents=True)


icu_csv_path = "icu_updated_v2.csv"
if not os.path.exists(icu_csv_path):
    raise FileNotFoundError(f"The file {icu_csv_path} does not exist. Please upload it.")

# Load ICU data
df = pd.read_csv(icu_csv_path, usecols=["GatewayName", "HR", "NIBP_Systolic", "NIBP_Diastolic", "SpO2", "RR", "Timestamp","GCS","Age"])

# NEWS Score Calculation
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
    
def calculate_apache(hr, systolic_bp, age, gcs):
    score = 0
    if hr < 40 or hr > 180:
        score += 4
    elif 40 <= hr < 55 or 140 <= hr <= 180:
        score += 3
    elif 55 <= hr < 70 or 110 <= hr < 140:
        score += 2
    if systolic_bp < 70 or systolic_bp > 200:
        score += 4
    elif 70 <= systolic_bp < 80 or 180 <= systolic_bp <= 200:
        score += 3
    elif 80 <= systolic_bp < 100 or 160 <= systolic_bp < 180:
        score += 2
    if gcs:
        score += (15 - gcs)
    if age > 75:
        score += 6
    elif 65 <= age <= 74:
        score += 5
    elif 55 <= age <= 64:
        score += 3
    elif 45 <= age <= 54:
        score += 2
    return score

df["APACHE_II_Score"] = df.apply(lambda row: calculate_apache(row["HR"], row["NIBP_Systolic"], row["Age"], row["GCS"]), axis=1)

# SAPS II Score Calculation
def calculate_saps(hr, systolic_bp, age, gcs):
    score = 0
    if hr > 160 or hr < 40:
        score += 8
    elif 40 <= hr < 70 or 120 <= hr <= 160:
        score += 5
    if systolic_bp < 70:
        score += 13
    elif 70 <= systolic_bp < 100:
        score += 5
    if gcs:
        score += (15 - gcs) * 2
    if age > 75:
        score += 6
    elif 65 <= age <= 74:
        score += 5
    elif 55 <= age <= 64:
        score += 3
    elif 45 <= age <= 54:
        score += 2
    return score

df["SAPS_II_Score"] = df.apply(lambda row: calculate_saps(row["HR"], row["NIBP_Systolic"], row["Age"], row["GCS"]), axis=1)

def recognize_speech_from_mic(recognizer, microphone):
    """Capture audio and convert it to text."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)
    
    response = {"success": True, "error": None, "transcription": None}
    
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    
    return response

def parse_vitals(transcription):
    """Extract patient vitals from transcribed speech using regex."""
    data = {}
    patient_id_match = re.search(r"patient id (\d+)", transcription, re.IGNORECASE)
    heart_rate_match = re.search(r"heart rate (\d+)", transcription, re.IGNORECASE)
    blood_pressure_match = re.search(r"blood pressure (\d+)[^\d]+(\d+)", transcription, re.IGNORECASE)
    temperature_match = re.search(r"temperature (\d+)", transcription, re.IGNORECASE)

    if patient_id_match:
        data["PATIENT ID"] = patient_id_match.group(1)
    if heart_rate_match:
        data["HEART RATE"] = heart_rate_match.group(1)
    if blood_pressure_match:
        data["BLOOD PRESSURE"] = f"{blood_pressure_match.group(1)}/{blood_pressure_match.group(2)}"
    if temperature_match:
        data["TEMPERATURE"] = temperature_match.group(1)
    
    return data

def save_to_csv(data, filename="patient_vitals.csv"):
    """Save extracted vitals data to CSV."""
    file_path = filename
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["PATIENT ID", "HEART RATE", "BLOOD PRESSURE", "TEMPERATURE"])
    
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(file_path, index=False)

df["Warning_Message"] = df["NEWS_Score"].apply(get_warning_message)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.sort_values(by=["GatewayName", "Timestamp"], inplace=True)

# Calculate time difference in seconds
df["Time_Diff"] = df.groupby("GatewayName")["Timestamp"].diff().dt.total_seconds()

# Calculate heart rate change
df["HR_Change"] = df.groupby("GatewayName")["HR"].diff()

# Initialize Condition column
df["Condition"] = "Normal"

# Apply Tachycardia and Bradycardia detection only if the time difference is ≤ 15 seconds
df.loc[(df["HR_Change"] > 25) & (df["Time_Diff"] <= 15), "Condition"] = "Tachycardia"
df.loc[(df["HR_Change"] < -15) & (df["Time_Diff"] <= 15), "Condition"] = "Bradycardia"

# Filter critical patients
critical_patients = df[df["Condition"] != "Normal"]

csv_agent = PythonAgent(
    model=Ollama(id="llama3.2"),
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



if 'patient_notes' not in st.session_state:
    st.session_state.patient_notes = {}
# Streamlit UI
def main():
    st.set_page_config(page_title="ICU Monitoring System", layout="wide")
    st.title("🚑 ICU Patient Monitoring & Alert System")

    # NEWS Score Table
    st.subheader("📊 NEWS Score")
    news_table = df[["GatewayName", "Timestamp", "NEWS_Score", "Warning_Message"]]
    gb = GridOptionsBuilder.from_dataframe(news_table)
    gb.configure_pagination()
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(news_table, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)

    # Add APACHE II & SAPS II Scores to Streamlit UI

    st.subheader("📊 APACHE II & SAPS II Score Analysis")
    apache_saps_table = df[["GatewayName", "Timestamp", "APACHE_II_Score", "SAPS_II_Score"]]
    gb = GridOptionsBuilder.from_dataframe(apache_saps_table)
    gb.configure_pagination()
    grid_options = gb.build()
    AgGrid(apache_saps_table, gridOptions=grid_options, height=300, fit_columns_on_grid_load=True)
    
    st.subheader("📊 NEWS Score Analysis ")
    # Gateway filter
    gateways = df["GatewayName"].unique()
    selected_gateway = st.selectbox("Select Gateway", gateways)
    filtered_df = df[df["GatewayName"] == selected_gateway]
    
    # Plot NEWS Score Spike Detection
    parameter = st.selectbox("Select Parameter for Spike Detection", ["NEWS_Score", "APACHE_II_Score", "SAPS_II_Score", "HR", "NIBP_Systolic", "RR"])

    if parameter in ["NEWS_Score", "APACHE_II_Score", "SAPS_II_Score"]:
       filtered_df = filtered_df[filtered_df[parameter] > 5]
       title = f"{parameter} Spikes"
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

    st.subheader("📝 Nurse Notes & Treatment History")
    patient_id = st.text_input("Enter Patient ID:")
    notes = st.text_area("Enter Treatment Notes or Problem Description:")

    if st.button("Save Notes"):
        if patient_id and notes:
            if patient_id in st.session_state.patient_notes:
                st.session_state.patient_notes[patient_id].append(notes)
            else:
                st.session_state.patient_notes[patient_id] = [notes]
            st.success("Notes saved successfully!")
        else:
            st.error("Please enter both Patient ID and Notes.")

    if st.button("View Notes"):
        if patient_id in st.session_state.patient_notes:
            st.subheader(f"Notes for Patient ID: {patient_id}")
            for i, note in enumerate(st.session_state.patient_notes[patient_id], 1):
                st.write(f"{i}. {note}")
        else:
            st.error("No notes found for this Patient ID.")

    # Voice-Based Data Entry
    st.subheader("🎙️ Voice-Based Nurse Data Entry")
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    if st.button("Capture Vitals via Speech"):
        response = recognize_speech_from_mic(recognizer, microphone)
        
        if response["success"]:
            transcription = response["transcription"]
            st.write(f"You said: {transcription}")
            vitals_data = parse_vitals(transcription)
            
            if vitals_data:
                st.write("Extracted Vitals:", vitals_data)
                save_to_csv(vitals_data)
                st.success("Data saved successfully.")
            else:
                st.error("Could not extract vitals. Please speak clearly.")
        else:
            st.error(f"Error: {response['error']}")

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
