import json
import re
import requests
import speech_recognition as sr
import pyttsx3
import streamlit as st

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text.strip()
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Please repeat.")
            return listen()
        except sr.RequestError:
            speak("There was an issue with the speech recognition service.")
            return None

def validate_input(prompt, pattern, error_message):
    while True:
        speak(prompt)
        response = listen()
        if response and re.match(pattern, response, re.IGNORECASE):
            return response
        else:
            speak(error_message)

def collect_patient_info():
    patient_data = {}
    
    patient_data["Name"] = validate_input(
        "What is the patient's name?", r"^[A-Za-z ]+$", "Please provide a valid name using only letters.")
    
    patient_data["Age"] = validate_input(
        "What is the patient's age?", r"^\d{1,3}$", "Please provide a valid age in numbers.")
    
    patient_data["Gender"] = validate_input(
        "What is the patient's gender?", r"^(male|female|other)$", "Please say Male, Female, or Other.")
    
    patient_data["UHID Number"] = validate_input(
        "Please provide the UHID number.", r"^\d{6,10}$", "Please provide a valid UHID number.")
    
    patient_data["Case Type"] = validate_input(
        "What is the case type?", r"^[A-Za-z ]+$", "Please provide a valid case type using only letters.")
    
    patient_data["Case Description"] = validate_input(
        "Please describe the case.", r"^.{10,}$", "Please provide a longer case description.")
    
    with open("patient_data.json", "w") as file:
        json.dump(patient_data, file, indent=4)
    
    api_url = "https://careworxdevapi-d3eddedhd2axdzf6.centralindia-01.azurewebsites.net/api/v6/IAPI_PatientCreation/add"  # Replace with actual API endpoint
    response = requests.post(api_url, json=patient_data)
    
    if response.status_code == 200:
        speak("Patient registration is complete and data has been sent successfully.")
        st.write("Patient data saved and sent to API:", patient_data)
    else:
        speak("There was an error sending the data to the API.")
        st.write("Error sending data to API:", response.text)

def main():
    st.title("Voice-Based Patient Registration System")
    st.write("Click the button below to start the registration process.")
    if st.button("Start Registration"):
        collect_patient_info()

if __name__ == "__main__":
    main()
