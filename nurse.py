import os
import subprocess
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.groq import Groq
import streamlit as st

# Load environment variables
load_dotenv()

# Set up base directory for temporary files
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

# Path to the ICU CSV file
local_csv_path = "icu.csv"

# Ensure the CSV file exists
if not os.path.exists(local_csv_path):
    raise FileNotFoundError(f"The file {local_csv_path} does not exist.")

# Load ICU data
icu_df = pd.read_csv(local_csv_path)

# Ensure required columns exist
required_columns = {"GatewayName", "Timestamp", "HR"}
if not required_columns.issubset(icu_df.columns):
    raise ValueError(f"Dataset is missing required columns: {required_columns - set(icu_df.columns)}")

# Define thresholds for tachycardia and bradycardia
TACHYCARDIA_THRESHOLD = 100
BRADYCARDIA_THRESHOLD = 60

# Identify critical patients
critical_patients = icu_df[(icu_df["HR"] > TACHYCARDIA_THRESHOLD) | (icu_df["HR"] < BRADYCARDIA_THRESHOLD)]

# Configure the AI Agent
python_agent = PythonAgent(
    model=Groq(id="mixtral-8x7b-32768"),
    base_dir=tmp,
    files=[
        CsvFile(
            path=local_csv_path,
            description="ICU dataset containing patient heart rate, timestamps, and GatewayName for monitoring.",
        )
    ],
    markdown=True,
    pip_install=True,
    show_tool_calls=True,
    allow_execution=True
)

# Streamlit UI
def main():
    st.title("ICU Patient Monitoring & Alert System")
    st.write("Analyze ICU data and detect critical heart rate conditions.")
    
    # Display critical patients
    if not critical_patients.empty:
        st.subheader("⚠️ Critical Patients Detected")
        st.dataframe(critical_patients[["GatewayName", "Timestamp", "HR"]])
    else:
        st.success("✅ No critical patients detected.")
    
    # User query input
    question = st.text_area("Enter your question about the ICU data:", placeholder="e.g., How many patients have bradycardia?")
    
    if st.button("Analyze Data"):
        if not question.strip():
            st.error("Please enter a valid question.")
            return
        
        try:
            with st.spinner("Processing your question..."):
                response = python_agent.run(question)
                st.markdown(response.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()