import os 
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


load_dotenv()


cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
tmp.mkdir(exist_ok=True, parents=True)


local_csv_path = "icu.csv"


if not os.path.exists(local_csv_path):
    st.error(f"The file {local_csv_path} does not exist.")
    st.stop()


icu_df = pd.read_csv(local_csv_path)


required_columns = {"GatewayName", "Timestamp", "HR"}
missing_columns = required_columns - set(icu_df.columns)
if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()




icu_df.sort_values(by=["GatewayName", "Timestamp"], inplace=True)

# Calculate ΔHR (change in heart rate)
icu_df["HR_Change"] = icu_df.groupby("GatewayName")["HR"].diff()

# Identify critical conditions based on HR change
icu_df["Condition"] = "Normal"
icu_df.loc[icu_df["HR_Change"] > 25, "Condition"] = "Tachycardia"
icu_df.loc[icu_df["HR_Change"] < -15, "Condition"] = "Bradycardia"

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
            header=dict(values=["Gateway Name", "Timestamp", "Heart Rate", "ΔHR", "Condition"],
                        fill_color='lightblue',
                        align='center',
                        font=dict(color='black', size=14)),
            cells=dict(values=[filtered_patients.GatewayName, 
                               filtered_patients.Timestamp, 
                               filtered_patients.HR, 
                               filtered_patients.HR_Change,
                               filtered_patients.Condition],
                       fill_color='white',
                       align='center',
                       font=dict(color='black', size=12)))
        ])
        st.plotly_chart(fig)
    else:
        st.success("No critical patients detected for the selected condition.")

    # User query input
    question = st.text_area("Enter your question about the ICU data:", placeholder="e.g., How many patients have bradycardia?")

    if st.button("Analyze Data"):
        if not question.strip():
            st.error("Please enter a valid question.")
            return

        if "tachycardia" in question.lower():
            tachycardia_count = (icu_df["Condition"] == "Tachycardia").sum()
            st.markdown(f"**{tachycardia_count} patients have tachycardia.**")
        elif "bradycardia" in question.lower():
            bradycardia_count = (icu_df["Condition"] == "Bradycardia").sum()
            st.markdown(f"**{bradycardia_count} patients have bradycardia.**")
        else:
            st.error("Sorry, I can't process that query.")

if __name__ == "__main__":
    main()
