import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Replace with your real LPR + classification logic
def extract_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plate_text = pytesseract.image_to_string(gray)
    return plate_text.strip()

def identify_state(plate_text):
    state_prefix = plate_text[:1].upper()
    state_map = {
        "A": "Perak",
        "B": "Selangor",
        "C": "Pahang",
        "D": "Kelantan",
        "EV": "Special Series",
        "F": "W.P. Putrajaya",
        "FFF": "Special Series",
        "GOLD": "Special Series",
        "H": "Taxi",
        "J": "Johor",
        "K": "Kedah",
        "KV": "Langkawi",
        "L": "W.P. Labuan",
        "LIMO": "Special Series",
        "M": "Melaka",
        "MADANI": "Special Series",
        "N": "Negeri Sembilan",
        "P": "Pulau Pinang",
        "PETRA": "Special Series",
        "Q": "Sarawak",
        "R": "Perlis",
        "S": "Sabah",
        "T": "Terengganu",
        "U": "Special Series",
        "V": "W.P. Kuala Lumpur",
        "VIP": "Special Series",
        "W": "W.P. Kuala Lumpur",
        "X": "Special Series",
        "Y": "Special Series",
        "Z": "Military"
    }
    return state_map.get(state_prefix, "Unknown")

def classify_vehicle(image):
    # Placeholder logic
    return "Car"

st.set_page_config(page_title="LPR & State ID System", layout="centered")

st.title("📸 License Plate Recognition (LPR) + State ID System")

if "images" not in st.session_state:
    st.session_state.images = []
if "results" not in st.session_state:
    st.session_state.results = []

# Upload Images
uploaded = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded:
    st.session_state.images = uploaded

# Process Button
if st.button("Start") and st.session_state.images:
    st.session_state.results.clear()
    for file in st.session_state.images:
        img = Image.open(file)
        img_np = np.array(img)
        plate_text = extract_plate_text(img_np)
        state = identify_state(plate_text)
        vehicle = classify_vehicle(img_np)

        st.session_state.results.append({
            "filename": file.name,
            "plate": plate_text,
            "state": state,
            "vehicle": vehicle,
            "image": img
        })

# Show Results
if st.session_state.results:
    for result in st.session_state.results:
        st.image(result["image"], caption=result["filename"], width=300)
        st.markdown(f"""
        **Plate:** `{result['plate']}`  
        **State:** `{result['state']}`  
        **Vehicle Type:** `{result['vehicle']}`  
        """)

# Clear Button
if st.button("Clear"):
    st.session_state.images = []
    st.session_state.results = []
    st.experimental_rerun()