import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re
from paddleocr import TextDetection

# Global variable to track OCR engine type
OCR_ENGINE = None

@st.cache_resource
def load_ocr_model():
    """Load OCR model with fallback options"""
    global OCR_ENGINE
    
    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        import paddleocr
        
        # Check PaddleOCR version and initialize accordingly
        ocr_model = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False
        )
        
        OCR_ENGINE = "paddleocr"
        return ocr_model, "paddleocr"
    except Exception as e:
        st.warning(f"⚠️ PaddleOCR failed to load: {str(e)}")
        return None, None

def extract_license_plate_text_correct(ocr_model, image):
    """
    Correct OCR function for dictionary-format results
    """
    try:
        # Get OCR results
        try:
            results = ocr_model.ocr(image, cls=True)
        except TypeError:
            results = ocr_model.ocr(image)
        
        print(f"OCR Results type: {type(results)}")
        
        # Handle different result formats
        if isinstance(results, dict):
            # Dictionary format - extract from 'res' key if it exists
            if 'res' in results:
                ocr_data = results['res']
            else:
                ocr_data = results
                
        elif isinstance(results, list) and len(results) > 0:
            # List format - might contain dictionary
            if isinstance(results[0], dict):
                if 'res' in results[0]:
                    ocr_data = results[0]['res']
                else:
                    ocr_data = results[0]
        else:
            print("Unexpected results format")
            return None
        
        # Extract text and scores from dictionary format
        if 'rec_texts' in ocr_data and 'rec_scores' in ocr_data:
            texts = ocr_data['rec_texts']
            scores = ocr_data['rec_scores']
            
            print(f"Found texts: {texts}")
            print(f"Found scores: {scores}")
            
            if not texts or len(texts) == 0:
                print("No text detected")
                return None
            
            # Find best text based on confidence and length
            best_text = None
            highest_score = 0
            
            for i, (text, score) in enumerate(zip(texts, scores)):
                print(f"Text {i}: '{text}' (confidence: {score:.3f})")
                
                # Basic filtering for license plates
                clean_text = text.strip()
                if len(clean_text) >= 3 and score > highest_score:
                    best_text = clean_text
                    highest_score = score
            
            if best_text:
                # Clean the text (remove spaces for license plates)
                final_text = best_text.replace(" ", "")
                print(f"Selected text: '{final_text}' with confidence {highest_score:.3f}")
                return final_text
            else:
                print("No suitable text found")
                return None
        else:
            print("No 'rec_texts' or 'rec_scores' found in results")
            print(f"Available keys: {list(ocr_data.keys())}")
            return None
            
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return None

def extract_plate_text(image, ocr_model, ocr_engine):
    """Extract text from license plate using available OCR engine"""
    if ocr_engine == "paddleocr" and ocr_model:
        return extract_license_plate_text_correct(ocr_model, image)
    else:
        st.error("❌ No OCR engine available!")
        return ""

def identify_state(plate_text):
    """Identify Malaysian state from license plate text"""
    if not plate_text:
        return "Unknown"
    
    clean_plate = plate_text.replace(' ', '').upper()
    
    state_map = {
        # Single letter states
        "A": "Perak",
        "B": "Selangor", 
        "C": "Pahang",
        "D": "Kelantan",
        "F": "W.P. Putrajaya",
        "J": "Johor",
        "K": "Kedah",
        "L": "W.P. Labuan",
        "M": "Melaka",
        "N": "Negeri Sembilan",
        "P": "Pulau Pinang",
        "Q": "Sarawak",
        "R": "Perlis",
        "S": "Sabah",
        "T": "Terengganu",
        "V": "W.P. Kuala Lumpur",
        "W": "W.P. Kuala Lumpur",
        "Z": "Military",
        
        # Multi-letter prefixes
        "KV": "Langkawi",
        "EV": "Special Series",
        "FFF": "Special Series", 
        "VIP": "Special Series",
        "GOLD": "Special Series",
        "LIMO": "Special Series",
        "MADANI": "Special Series",
        "PETRA": "Special Series",
        "U": "Special Series",
        "X": "Special Series",
        "Y": "Special Series",
        "H": "Taxi"
    }
    
    for prefix in sorted(state_map.keys(), key=len, reverse=True):
        if clean_plate.startswith(prefix):
            return state_map[prefix]
    
    first_char = clean_plate[0] if clean_plate else ""
    return state_map.get(first_char, "Unknown")

def classify_vehicle(image):
    """Classify vehicle type (placeholder)"""
    return "Car"

# Streamlit App Configuration
st.set_page_config(
    page_title="Malaysian LPR System", 
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Malaysian License Plate Recognition System")
st.markdown("Upload images of Malaysian license plates to extract text and identify the state of registration.")

# Initialize OCR - this will be cached
ocr_model, ocr_engine = load_ocr_model()

# Update global variable
OCR_ENGINE = ocr_engine

# Check if OCR is available
if ocr_model is None:
    st.error("❌ Failed to initialize OCR engine. Please check the installation.")
    st.stop()
else:
    st.success("✅ Using PaddleOCR engine")

# Initialize session state
if "images" not in st.session_state:
    st.session_state.images = []
if "results" not in st.session_state:
    st.session_state.results = []

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown(f"""
    **Current OCR Engine:** {OCR_ENGINE.upper() if OCR_ENGINE else 'None'}
    
    **Supported States:**
    - All Malaysian states
    - Federal territories
    - Special series plates
    - Military plates
    - Taxi plates
    
    **Tips for better results:**
    - Use clear, well-lit images
    - Ensure license plate is visible
    - Avoid blurry or angled shots
    """)

# File Upload Section
st.header("📁 Upload Images")
uploaded_files = st.file_uploader(
    "Choose license plate images", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True,
    help="Upload one or more images containing Malaysian license plates"
)

if uploaded_files:
    st.session_state.images = uploaded_files
    st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")

# Processing Section
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Process Images", type="primary", disabled=not st.session_state.images):
        if st.session_state.images:
            st.session_state.results.clear()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(st.session_state.images):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    img = Image.open(uploaded_file)
                    img_np = np.array(img)
                    
                    plate_text = extract_plate_text(img_np, ocr_model, ocr_engine)
                    state = identify_state(plate_text)
                    vehicle_type = classify_vehicle(img_np)
                    
                    st.session_state.results.append({
                        "filename": uploaded_file.name,
                        "plate_text": plate_text,
                        "state": state,
                        "vehicle_type": vehicle_type,
                        "image": img
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(st.session_state.images))
            
            status_text.text("✅ Processing completed!")
            st.balloons()

with col2:
    if st.button("🗑️ Clear All", type="secondary"):
        st.session_state.images = []
        st.session_state.results = []
        st.rerun()

# Results Section
if st.session_state.results:
    st.header("📊 Results")
    
    for i, result in enumerate(st.session_state.results, 1):
        with st.expander(f"🖼️ Result {i}: {result['filename']}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(result["image"], caption=result["filename"], use_container_width=True)
            
            with col2:
                st.markdown("### Extracted Information")
                
                info_html = f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #1f77b4;">🔢 License Plate</h4>
                    <p style="font-size: 18px; font-weight: bold; margin: 0; color: #333;">
                        {result['plate_text'] if result['plate_text'] else 'Not detected'}
                    </p>
                </div>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #2e8b57;">📍 State/Region</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0; color: #333;">
                        {result['state']}
                    </p>
                </div>
                
                <div style="background-color: #fff0e6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0; color: #ff8c00;">🚗 Vehicle Type</h4>
                    <p style="font-size: 16px; font-weight: bold; margin: 0; color: #333;">
                        {result['vehicle_type']}
                    </p>
                </div>
                """
                st.markdown(info_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🚗 Malaysian License Plate Recognition System | "
    f"OCR Engine: {OCR_ENGINE.upper() if OCR_ENGINE else 'None'} | "
    f"Built with Streamlit"
    f"</div>", 
    unsafe_allow_html=True
)