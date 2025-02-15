import streamlit as st
import speech_recognition as sr
import pandas as pd
import re

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
    """Extract patient vitals from the transcribed text using regex for better accuracy."""
    data = {}
    
    # Regular expressions to extract numerical values
    patient_id_match = re.search(r"patient id (?:is )?(\d+)", transcription, re.IGNORECASE)
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
    """Save the extracted data to a CSV file with correct column names."""
    file_path = filename
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["PATIENT ID", "HEART RATE", "BLOOD PRESSURE", "TEMPERATURE"])
    
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(file_path, index=False)

def main():
    st.title("Patient Vitals Speech Recognition")
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    if st.button("Capture Vitals via Speech"):
        response = recognize_speech_from_mic(recognizer, microphone)
        
        if response["success"]:
            transcription = response["transcription"]
            st.write(f"You said: {transcription}")
            vitals_data = parse_vitals(transcription)
            
            if vitals_data:
                st.write("Extracted Vitals:")
                st.write(vitals_data)
                save_to_csv(vitals_data)
                st.success("Data saved to CSV.")
            else:
                st.error("Could not extract vitals. Please speak clearly.")
        else:
            st.error(f"Error: {response['error']}")

if __name__ == "__main__":
    main()
