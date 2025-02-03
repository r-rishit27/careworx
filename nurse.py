import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Set up base directory for temporary files
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
tmp.mkdir(exist_ok=True, parents=True)

# Path to the ICU CSV file
local_csv_path = "icu.csv"

# Ensure the CSV file exists
if not os.path.exists(local_csv_path):
    st.error(f"The file {local_csv_path} does not exist.")
    st.stop()

# Load ICU data
icu_df = pd.read_csv(local_csv_path)

# Ensure required columns exist
required_columns = {"GatewayName", "Timestamp", "HR"}
missing_columns = required_columns - set(icu_df.columns)
if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()

# Define thresholds for tachycardia and bradycardia
TACHYCARDIA_THRESHOLD = 100
BRADYCARDIA_THRESHOLD = 60

# Identify critical patients and classify their condition
icu_df["Condition"] = "Normal"
icu_df.loc[icu_df["HR"] > TACHYCARDIA_THRESHOLD, "Condition"] = "Tachycardia"
icu_df.loc[icu_df["HR"] < BRADYCARDIA_THRESHOLD, "Condition"] = "Bradycardia"

critical_patients = icu_df[icu_df["Condition"] != "Normal"]

# Streamlit UI
def main():
    st.title("ICU Patient Monitoring & Alert System")
    st.write("Analyze ICU data and detect critical heart rate conditions.")

    # Filter selection
    condition_filter = st.selectbox("Select Condition", ["All", "Tachycardia", "Bradycardia"], index=0)

    # Display critical patients based on filter
    if condition_filter == "All":
        filtered_patients = critical_patients
    else:
        filtered_patients = critical_patients[critical_patients["Condition"] == condition_filter]

    if not filtered_patients.empty:
        st.subheader(f"⚠️ Critical Patients Detected ({condition_filter})")
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Gateway Name", "Timestamp", "Heart Rate", "Condition"],
                        fill_color='lightblue',
                        align='center',
                        font=dict(color='black', size=14)),
            cells=dict(values=[filtered_patients.GatewayName, 
                               filtered_patients.Timestamp, 
                               filtered_patients.HR, 
                               filtered_patients.Condition],
                       fill_color='white',
                       align='center',
                       font=dict(color='black', size=12)))
        ])
        st.plotly_chart(fig)
    else:
        st.success(" No critical patients detected for the selected condition.")

    # User query input
    question = st.text_area("Enter your question about the ICU data:", placeholder="e.g., How many patients have bradycardia?")

    if st.button("Analyze Data"):
        if not question.strip():
            st.error("Please enter a valid question.")
            return

        if "tachycardia" in question.lower():
            tachycardia_count = (icu_df["HR"] > TACHYCARDIA_THRESHOLD).sum()
            st.markdown(f"**{tachycardia_count} patients have tachycardia.**")
        elif "bradycardia" in question.lower():
            bradycardia_count = (icu_df["HR"] < BRADYCARDIA_THRESHOLD).sum()
            st.markdown(f"**{bradycardia_count} patients have bradycardia.**")
        else:
            st.error("Sorry, I can't process that query.")

if __name__ == "__main__":
    main()
