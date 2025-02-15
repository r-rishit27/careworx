import streamlit as st

if 'patient_notes' not in st.session_state:
    st.session_state.patient_notes = {}

st.title("Patient Treatment Notes Recorder")


st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option", ["Add Notes", "View Notes"])

# Function to add notes for a patient
def add_notes():
    st.header("Add Patient Notes")
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

# Function to view notes for a patient
def view_notes():
    st.header("View Patient Notes")
    patient_id = st.text_input("Enter Patient ID to View Notes:")

    if st.button("View Notes"):
        if patient_id in st.session_state.patient_notes:
            st.subheader(f"Notes for Patient ID: {patient_id}")
            for i, note in enumerate(st.session_state.patient_notes[patient_id], 1):
                st.write(f"{i}. {note}")
        else:
            st.error("No notes found for this Patient ID.")

# Display the selected option
if option == "Add Notes":
    add_notes()
elif option == "View Notes":
    view_notes()