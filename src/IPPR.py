import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re
import pywt
import logging
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            use_textline_orientation=False,
            show_log=False  # Reduce verbose output
        )
        
        OCR_ENGINE = "paddleocr"
        st.success("✅ PaddleOCR loaded successfully!")
        return ocr_model, "paddleocr"
    except Exception as e:
        st.warning(f"⚠️ PaddleOCR failed to load: {str(e)}")
        logger.error(f"PaddleOCR loading failed: {e}")
        return None, None

def detect_license_plate_regions(image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect license plate regions using multiple preprocessing approaches
    
    Args:
        image: Input grayscale image (NOT binary)
        
    Returns:
        List of tuples (x, y, w, h, area) for detected plate candidates
    """
    all_candidates = []
    
    # Method 1: Adaptive thresholding
    try:
        adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        candidates1 = find_plate_candidates_from_binary(adaptive_thresh)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "adaptive") for c in candidates1])
    except:
        pass
    
    # Method 2: Dark region detection (license plates are typically dark)
    try:
        # Invert image to find dark regions as white
        inverted = cv2.bitwise_not(image)
        _, dark_thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2 = find_plate_candidates_from_binary(dark_thresh)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "dark") for c in candidates2])
    except:
        pass
    
    # Method 2b: Standard Otsu thresholding
    try:
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2b = find_plate_candidates_from_binary(otsu_thresh)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "otsu") for c in candidates2b])
    except:
        pass
    
    # Method 3: Enhanced edge detection for license plates
    try:
        # Use bilateral filter before edge detection to reduce noise but keep edges
        bilateral = cv2.bilateralFilter(image, 11, 17, 17)
        edges = cv2.Canny(bilateral, 30, 100)
        
        # Use wider horizontal kernel to connect license plate characters
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))  # Wider to connect "AAT" and "40"
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
        
        # Additional horizontal dilation to connect characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))  # Wider horizontal connection
        morph = cv2.dilate(morph, kernel_dilate, iterations=2)
        
        candidates3 = find_plate_candidates_from_binary(morph)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "edge") for c in candidates3])
    except:
        pass
    
    # Method 4: License plate specific detection (dark regions with high contrast)
    try:
        # Look for regions that are darker than average but have high local contrast
        mean_intensity = np.mean(image)
        
        # Create mask for dark regions
        dark_mask = image < (mean_intensity * 0.7)
        dark_mask = dark_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to connect license plate characters
        # Use wider horizontal kernel to connect characters on the same plate
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Wider to connect characters
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_connect)
        
        # Clean up noise with smaller kernel
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_clean)
        
        candidates4 = find_plate_candidates_from_binary(dark_mask)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "contrast") for c in candidates4])
    except:
        pass
    
    # Method 5: Light text on dark background (bus plates)
    try:
        # Look for bright regions (light text) on darker backgrounds
        mean_intensity = np.mean(image)
        
        # Create mask for bright regions (light text) - more aggressive threshold
        bright_mask = image > (mean_intensity * 1.1)  # Lower threshold to catch more text
        bright_mask = bright_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to connect light text characters
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 6))  # Even wider for bus plates
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_connect)
        
        # Additional dilation to ensure text connection
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        bright_mask = cv2.dilate(bright_mask, kernel_dilate, iterations=1)
        
        # Clean up with opening
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_clean)
        
        candidates5 = find_plate_candidates_from_binary(bright_mask)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "bright") for c in candidates5])
    except:
        pass
    
    # Method 6: Bus-specific license plate detection
    try:
        height, width = image.shape[:2]
        
        # Focus on lower portion of image where bus plates typically are
        lower_third = image[height//3:, :]  # Bottom 2/3 of image
        
        # Use very aggressive thresholding for bus plates
        _, bus_thresh = cv2.threshold(lower_third, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Very wide morphological operations for bus text
        kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))  # Extra wide for bus plates
        bus_processed = cv2.morphologyEx(bus_thresh, cv2.MORPH_CLOSE, kernel_wide)
        
        # Additional connection with dilation
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        bus_processed = cv2.dilate(bus_processed, kernel_dilate, iterations=2)
        
        # Clean up small noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
        bus_processed = cv2.morphologyEx(bus_processed, cv2.MORPH_OPEN, kernel_clean)
        
        # Find candidates in the processed lower portion
        bus_candidates = find_plate_candidates_from_binary(bus_processed)
        
        # Adjust y-coordinates back to full image coordinates
        adjusted_bus_candidates = []
        for x, y, w, h, area in bus_candidates:
            adjusted_y = y + height//3  # Add offset for lower third
            adjusted_bus_candidates.append((x, adjusted_y, w, h, area, "bus"))
        
        all_candidates.extend(adjusted_bus_candidates)
    except:
        pass
    
    # Remove duplicates (similar positions)
    unique_candidates = []
    for candidate in all_candidates:
        x, y, w, h, area, method = candidate
        is_duplicate = False
        for existing in unique_candidates:
            ex, ey, ew, eh, ea, em = existing
            # Check if centers are close (within 20 pixels)
            if abs((x + w/2) - (ex + ew/2)) < 20 and abs((y + h/2) - (ey + eh/2)) < 20:
                is_duplicate = True
                # Keep the larger area
                if area > ea:
                    unique_candidates.remove(existing)
                    unique_candidates.append(candidate)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Convert back to original format and sort
    final_candidates = [(x, y, w, h, area) for x, y, w, h, area, method in unique_candidates]
    final_candidates.sort(key=lambda x: x[4], reverse=True)
    return final_candidates[:10]

def find_plate_candidates_from_binary(binary_image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Helper function to find candidates from binary image"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000 or area > 50000:  # Target whole license plates, not fragments
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w == 0:
            continue
            
        aspect_ratio = w / h
        if not (1.5 <= aspect_ratio <= 5.0):  # Standard license plate aspect ratios
            continue
            
        # Calculate quality metrics
        extent = area / (w * h) if (w * h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Check if contour is approximately rectangular (license plate characteristic)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Approximate contour to check for rectangular shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter for license plate characteristics
        is_rectangular = len(approx) >= 4 and len(approx) <= 8  # Roughly rectangular
        good_circularity = 0.1 <= circularity <= 0.9  # Not too circular, not too irregular
        
        if (extent > 0.4 and solidity > 0.65 and 
            w > 60 and h > 25 and 
            is_rectangular and good_circularity):
            candidates.append((x, y, w, h, area))
    
    return candidates

def enhance_plate_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Apply multiple enhancement techniques to detected plate region
    
    Args:
        image: Input grayscale image
        x, y, w, h: Region of interest coordinates
        
    Returns:
        Enhanced plate region using multiple processing techniques
    """
    # Validate input coordinates
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        logger.warning(f"Invalid coordinates: x={x}, y={y}, w={w}, h={h}")
        return np.zeros((max(h, 50), max(w, 100)), dtype=np.uint8)
    
    # Extract region of interest with bounds checking
    height, width = image.shape[:2]
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # Ensure width and height don't exceed image bounds
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    
    # Recalculate actual width and height
    w = max(1, x_end - x)
    h = max(1, y_end - y)
    
    roi = image[y:y+h, x:x+w]
    
    # Handle empty or invalid ROI
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        logger.warning(f"Empty ROI extracted: roi.shape={roi.shape if roi.size > 0 else 'empty'}")
        # Return proportional fallback based on aspect ratio
        aspect_ratio = w / h if h > 0 else 2.0
        fallback_h = 50
        fallback_w = int(fallback_h * aspect_ratio)
        return np.zeros((fallback_h, fallback_w), dtype=np.uint8)
    
    # Start with a copy of the ROI
    enhanced_roi = roi.copy()
    
    # Basic enhancement: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_roi = clahe.apply(enhanced_roi)
        
        # Apply bilateral filtering for noise reduction
        if enhanced_roi.shape[0] >= 5 and enhanced_roi.shape[1] >= 5:
            enhanced_roi = cv2.bilateralFilter(enhanced_roi, 5, 50, 50)
        
        # Simple contrast stretching
        min_val, max_val = np.min(enhanced_roi), np.max(enhanced_roi)
        if max_val > min_val:
            enhanced_roi = ((enhanced_roi - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
        
    except Exception as e:
        logger.warning(f"Enhancement failed, using original ROI: {e}")
        enhanced_roi = roi.copy()
    
    return enhanced_roi

def process_image_through_phases(img_np: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, int, int, float]]]:
    """
    Process image through all 9 phases and return processed images and plate candidates
    
    Args:
        img_np: Input image as numpy array
        
    Returns:
        Tuple of (phases_dict, plate_candidates)
    """
    phases = {}
    
    # Convert to grayscale for processing
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
        
    phases['original'] = img_np
    phases['grayscale'] = gray
    
    # Phase 2: Image Enhancement
    enhanced = cv2.equalizeHist(gray)
    gamma = 1.2
    gamma_corrected = np.array(255 * (enhanced / 255) ** gamma, dtype='uint8')
    phases['enhanced'] = gamma_corrected
    
    # Phase 3: Image Restoration
    restored = cv2.bilateralFilter(gamma_corrected, 11, 17, 17)
    phases['restored'] = restored
    
    # Phase 4: Color Image Processing
    if len(img_np.shape) == 3:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2]
    else:
        value_channel = gray.copy()
    phases['color_processed'] = value_channel
    
    # Phase 5: Wavelet Transform
    try:
        coeffs2 = pywt.dwt2(restored, 'db4')
        LL, (LH, HL, HH) = coeffs2
        detail_combined = np.sqrt(LH**2 + HL**2 + HH**2)
        
        # Normalize to prevent division by zero
        detail_min, detail_max = np.min(detail_combined), np.max(detail_combined)
        if detail_max > detail_min:
            detail_norm = np.uint8(255 * (detail_combined - detail_min) / (detail_max - detail_min))
        else:
            detail_norm = np.zeros_like(detail_combined, dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Wavelet transform failed: {e}")
        detail_norm = restored.copy()
    
    phases['wavelet'] = detail_norm
    
    # Phase 6: Image Compression
    h_orig, w_orig = restored.shape
    # Ensure minimum size to prevent errors
    new_w, new_h = max(1, w_orig//4), max(1, h_orig//4)
    compressed = cv2.resize(restored, (new_w, new_h))
    decompressed = cv2.resize(compressed, (w_orig, h_orig))
    phases['compressed'] = decompressed
    
    # Phase 7: Morphological Processing
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, kernel_rect)
    morph_grad = cv2.morphologyEx(morph_close, cv2.MORPH_GRADIENT, kernel_rect)
    phases['morphological'] = morph_grad
    
    # Phase 8: Segmentation
    adaptive_thresh = cv2.adaptiveThreshold(
        restored, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    phases['segmented'] = adaptive_thresh
    
    # Phase 9: Representation & Description
    # Try detection on multiple phases to find best results
    candidates_enhanced = detect_license_plate_regions(enhanced)
    candidates_restored = detect_license_plate_regions(restored) 
    candidates_morph = detect_license_plate_regions(morph_grad)
    
    # Combine all candidates and remove duplicates
    all_phase_candidates = []
    for candidates in [candidates_enhanced, candidates_restored, candidates_morph]:
        all_phase_candidates.extend(candidates)
    
    # Remove duplicates (similar positions)
    unique_candidates = []
    for candidate in all_phase_candidates:
        x, y, w, h, area = candidate
        is_duplicate = False
        for existing in unique_candidates:
            ex, ey, ew, eh, ea = existing
            # Check if centers are close (within 30 pixels)
            if abs((x + w/2) - (ex + ew/2)) < 30 and abs((y + h/2) - (ey + eh/2)) < 30:
                is_duplicate = True
                # Keep the larger area
                if area > ea:
                    unique_candidates.remove(existing)
                    unique_candidates.append(candidate)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Sort by intelligent scoring for license plates
    def license_plate_score(candidate):
        if len(candidate) == 6:
            x, y, w, h, area, method = candidate
        else:
            x, y, w, h, area = candidate
            method = "unknown"
        
        # Get image dimensions for position scoring
        img_height, img_width = enhanced.shape[:2]
        
        # Base score from area
        area_score = area
        
        # Extreme penalty for being at image edges (likely false detections)
        edge_penalty = 1.0
        if (y <= 5 or x <= 5 or  # Very close to edges
            (y + h) >= (img_height - 5) or (x + w) >= (img_width - 5) or  # Near opposite edges
            y < 30 or x < 30 or  # Regular edge detection
            (y + h) > (img_height - 30) or (x + w) > (img_width - 30)):  # Regular far edges
            edge_penalty = 0.001  # Even more extreme penalty (99.9% reduction)
        
        # Prefer license plates in lower 2/3 of image (where cars are)
        position_score = 1.0
        center_y = y + h/2
        if center_y > img_height * 0.3:  # Lower 70% of image
            position_score = 2.0  # Strong bonus for car area
        if center_y > img_height * 0.7:  # Bottom 30% of image
            position_score = 1.5  # Still good but slightly lower
        
        # Prefer reasonable license plate sizes (not too huge)
        size_score = 1.0
        if area > 25000:  # Very large areas are likely false positives
            size_score = 0.2
        elif area > 15000:  # Large areas are suspicious
            size_score = 0.4
        elif 5000 <= area <= 15000:  # Good size range for license plates
            size_score = 1.5
        elif 2000 <= area <= 5000:  # Smaller plates (like AAT40)
            size_score = 1.8  # Higher bonus for smaller realistic plates
        elif 1000 <= area <= 2000:  # Very small but possible
            size_score = 1.2
        
        # Aspect ratio preference
        aspect_ratio = w / h
        aspect_score = 1.0
        if 3.0 <= aspect_ratio <= 4.5:  # Ideal license plate aspect ratio
            aspect_score = 1.2
        elif 2.0 <= aspect_ratio <= 5.0:  # Acceptable range
            aspect_score = 1.0
        else:
            aspect_score = 0.8
        
        # Method-specific bonuses
        method_score = 1.0
        if method == "bus":  # Bus-specific detection gets highest priority
            method_score = 3.0
        elif method == "bright":  # Light text detection for buses
            method_score = 2.5
        elif method in ["dark", "contrast"]:  # Good for regular license plates
            method_score = 1.5
        
        return area_score * edge_penalty * position_score * size_score * aspect_score * method_score
    
    unique_candidates.sort(key=license_plate_score, reverse=True)
    plate_candidates = unique_candidates[:10]
    
    # Draw results on original image
    result_img = img_np.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for i, (x, y, w, h, area) in enumerate(plate_candidates):
        color = colors[i % len(colors)]
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(result_img, f'Candidate {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    phases['detection_result'] = result_img
    
    return phases, plate_candidates

def extract_license_plate_text_correct(ocr_model, image: np.ndarray) -> Optional[str]:
    """
    OCR function for text extraction from license plates
    
    Args:
        ocr_model: Loaded OCR model
        image: Input image for text extraction
        
    Returns:
        Extracted text or None if extraction failed
    """
    if ocr_model is None:
        return None
        
    try:
        # Ensure image is in correct format
        if len(image.shape) == 3:
            # Convert RGB to BGR for PaddleOCR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get OCR results
        try:
            results = ocr_model.ocr(image, cls=True)
        except TypeError:
            results = ocr_model.ocr(image)
        
        # Handle PaddleOCR results format
        if isinstance(results, list) and len(results) > 0:
            # Standard PaddleOCR format: list of list of [bbox, (text, confidence)]
            if results[0] is not None:
                texts_and_scores = []
                for detection in results[0]:
                    if len(detection) >= 2:
                        text, confidence = detection[1]
                        texts_and_scores.append((text, confidence))
                
                if texts_and_scores:
                    # Find best text based on confidence and length
                    best_text = None
                    highest_score = 0
                    
                    for text, score in texts_and_scores:
                        clean_text = text.strip()
                        if len(clean_text) >= 3 and score > highest_score:
                            best_text = clean_text
                            highest_score = score
                    
                    if best_text:
                        # Clean the text (remove spaces, convert to uppercase)
                        final_text = re.sub(r'[^A-Z0-9]', '', best_text.upper())
                        return final_text if len(final_text) >= 3 else None
        
        return None
            
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        st.error(f"OCR Error: {str(e)}")
        return None

def extract_plate_text(image: np.ndarray, ocr_model, ocr_engine: str) -> str:
    """Extract text from license plate using available OCR engine"""
    if ocr_engine == "paddleocr" and ocr_model:
        result = extract_license_plate_text_correct(ocr_model, image)
        return result if result else ""
    else:
        return ""

def identify_state(plate_text: str) -> str:
    """Identify Malaysian state from license plate text"""
    if not plate_text:
        return "Unknown"
    
    clean_plate = plate_text.replace(' ', '').upper()
    
    state_map = {
        # Single letter states
        "A": "Perak", "B": "Selangor", "C": "Pahang", "D": "Kelantan",
        "F": "W.P. Putrajaya", "J": "Johor", "K": "Kedah", "L": "W.P. Labuan",
        "M": "Melaka", "N": "Negeri Sembilan", "P": "Pulau Pinang", "Q": "Sarawak",
        "R": "Perlis", "S": "Sabah", "T": "Terengganu", "V": "W.P. Kuala Lumpur",
        "W": "W.P. Kuala Lumpur", "Z": "Military",
        
        # Multi-letter prefixes
        "KV": "Langkawi", "EV": "Special Series", "FFF": "Special Series",
        "VIP": "Special Series", "GOLD": "Special Series", "LIMO": "Special Series",
        "MADANI": "Special Series", "PETRA": "Special Series",
        "U": "Special Series", "X": "Special Series", "Y": "Special Series",
        "H": "Taxi"
    }
    
    # Check multi-letter prefixes first (longer matches)
    for prefix in sorted(state_map.keys(), key=len, reverse=True):
        if clean_plate.startswith(prefix):
            return state_map[prefix]
    
    # Default to unknown
    return "Unknown"

def classify_vehicle(image: np.ndarray) -> str:
    """Classify vehicle type (placeholder implementation)"""
    # This is a placeholder - you can implement actual vehicle classification here
    # For now, just return "Car" as default
    return "Car"

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error(f"File {uploaded_file.name} is too large (max 10MB)")
        return False
    
    return True

# Streamlit App Configuration
st.set_page_config(
    page_title="Malaysian LPR System", 
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #2e8b57);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚗 Malaysian License Plate Recognition System</h1><p>Complete system with 9-phase image processing and OCR recognition</p></div>', unsafe_allow_html=True)

# Initialize OCR
with st.spinner("Loading OCR model..."):
    ocr_model, ocr_engine = load_ocr_model()

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Processing mode selection
    processing_mode = st.radio(
        "Select Processing Mode:",
        ["Complete Analysis", "Quick Recognition", "Phase-by-Phase View"],
        help="Choose how detailed you want the analysis to be"
    )
    
    st.header("ℹ️ System Information")
    st.markdown(f"""
    **OCR Engine:** {OCR_ENGINE.upper() if OCR_ENGINE else 'None Available'}
    **Status:** {'🟢 Ready' if OCR_ENGINE else '🔴 OCR Not Available'}
    
    **Processing Phases:**
    1. 📥 Image Acquisition
    2. ✨ Image Enhancement
    3. 🔧 Image Restoration
    4. 🌈 Color Processing
    5. 🌊 Wavelet Transform
    6. 📦 Compression Handling
    7. 🧱 Morphological Processing
    8. 🧩 Segmentation
    9. 🧬 Representation & Description
    10. 🔤 OCR Recognition
    """)
    
    # System requirements
    st.header("📋 Requirements")
    st.markdown("""
    - **Image formats:** PNG, JPG, JPEG
    - **Max file size:** 10MB per image
    - **Recommended:** Clear, well-lit images
    - **Best results:** Front-facing plates
    """)

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "🔍 Phase Analysis", "📊 Results", "ℹ️ Help"])

with tab1:
    st.header("Image Upload and Processing")
    
    # File uploader with validation
    uploaded_files = st.file_uploader(
        "Choose license plate images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        help="Upload one or more images containing Malaysian license plates (max 10MB each)"
    )
    
    if uploaded_files:
        # Validate all files
        valid_files = [f for f in uploaded_files if validate_uploaded_file(f)]
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} valid image(s) uploaded successfully!")
            
            # Show file details
            with st.expander("📋 File Details", expanded=False):
                for file in valid_files:
                    st.write(f"• **{file.name}** ({file.size / 1024:.1f} KB)")
            
            # Initialize session state
            if "processing_results" not in st.session_state:
                st.session_state.processing_results = []
            
            # Processing options
            col1, col2 = st.columns([3, 1])
            with col1:
                process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
            with col2:
                clear_button = st.button("🗑️ Clear Results", use_container_width=True)
            
            if clear_button:
                st.session_state.processing_results.clear()
                st.success("Results cleared!")
                st.rerun()
            
            if process_button:
                st.session_state.processing_results.clear()
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                
                for i, uploaded_file in enumerate(valid_files):
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(valid_files)})")
                    
                    try:
                        # Load and validate image
                        img = Image.open(uploaded_file)
                        img_np = np.array(img)
                        
                        if img_np.size == 0:
                            st.error(f"Invalid image: {uploaded_file.name}")
                            continue
                        
                        # Process through all phases
                        with st.spinner(f"Processing phases for {uploaded_file.name}..."):
                            phases, plate_candidates = process_image_through_phases(img_np)
                        
                        # Extract text from best candidate if available
                        plate_text = ""
                        if plate_candidates and ocr_model:
                            # Use the best candidate (first one)
                            x, y, w, h, _ = plate_candidates[0]
                            plate_roi = enhance_plate_region(phases['restored'], x, y, w, h)
                            plate_text = extract_plate_text(plate_roi, ocr_model, ocr_engine)
                        
                        # Identify state and vehicle type
                        state = identify_state(plate_text)
                        vehicle_type = classify_vehicle(img_np)
                        
                        # Store results
                        result_data = {
                            "filename": uploaded_file.name,
                            "phases": phases,
                            "plate_candidates": plate_candidates,
                            "plate_text": plate_text,
                            "state": state,
                            "vehicle_type": vehicle_type,
                            "original_image": img,
                        }
                        
                        st.session_state.processing_results.append(result_data)
                        
                        # Show quick preview
                        with results_container.container():
                            st.success(f"✅ Processed: {uploaded_file.name}")
                            if plate_text:
                                st.info(f"🔢 Detected: **{plate_text}** ({state})")
                            else:
                                st.warning("🔍 No text detected")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        logger.error(f"Processing error for {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(valid_files))
                
                st.balloons()
        else:
            st.error("❌ No valid files uploaded. Please check file sizes and formats.")

with tab2:
    st.header("🔍 Image Processing Phase Analysis")
    
    if "processing_results" in st.session_state and st.session_state.processing_results:
        # Select image for phase analysis
        selected_idx = st.selectbox(
            "Select image for phase analysis:",
            range(len(st.session_state.processing_results)),
            format_func=lambda x: st.session_state.processing_results[x]["filename"]
        )
        
        if selected_idx is not None:
            result = st.session_state.processing_results[selected_idx]
            phases = result["phases"]
            
            st.subheader(f"📊 Phase Analysis: {result['filename']}")
            
            # Phase information
            phase_info = [
                ("original", "📥 Phase 1: Image Acquisition", "Original input image"),
                ("enhanced", "✨ Phase 2: Image Enhancement", "Histogram equalization + gamma correction"),
                ("restored", "🔧 Phase 3: Image Restoration", "Bilateral filtering for noise reduction"),
                ("color_processed", "🌈 Phase 4: Color Processing", "HSV Value channel extraction"),
                ("wavelet", "🌊 Phase 5: Wavelet Transform", "Detail coefficients highlighting edges"),
                ("compressed", "📦 Phase 6: Compression Effects", "Compression simulation"),
                ("morphological", "🧱 Phase 7: Morphological Processing", "Gradient to enhance boundaries"),
                ("segmented", "🧩 Phase 8: Segmentation", "Adaptive thresholding"),
                ("detection_result", "🧬 Phase 9: Detection Result", "License plate detection")
            ]
            
            # Display phases in a grid
            for i in range(0, len(phase_info), 2):
                col1, col2 = st.columns(2)
                
                # Left column
                if i < len(phase_info):
                    key, title, description = phase_info[i]
                    with col1:
                        if key in phases:
                            if key in ["original", "detection_result"]:
                                st.image(phases[key], caption=title, use_container_width=True)
                            else:
                                st.image(phases[key], caption=title, channels="GRAY", use_container_width=True)
                            st.markdown(f"*{description}*")
                            st.markdown("---")
                
                # Right column
                if i + 1 < len(phase_info):
                    key, title, description = phase_info[i + 1]
                    with col2:
                        if key in phases:
                            if key in ["original", "detection_result"]:
                                st.image(phases[key], caption=title, use_container_width=True)
                            else:
                                st.image(phases[key], caption=title, channels="GRAY", use_container_width=True)
                            st.markdown(f"*{description}*")
                            st.markdown("---")
    else:
        st.info("👆 Please upload and process images first to view phase analysis.")

with tab3:
    st.header("📊 Recognition Results")
    
    if "processing_results" in st.session_state and st.session_state.processing_results:
        # Summary statistics first
        st.subheader("📈 Processing Summary")
        
        total_processed = len(st.session_state.processing_results)
        successful_detections = sum(1 for r in st.session_state.processing_results if r["plate_text"])
        total_candidates = sum(len(r["plate_candidates"]) for r in st.session_state.processing_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Images Processed", total_processed)
        
        with col2:
            st.metric("Successful Detections", successful_detections)
        
        with col3:
            success_rate = (successful_detections / total_processed * 100) if total_processed > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        st.markdown("---")
        
        # Detailed results
        for i, result in enumerate(st.session_state.processing_results):
            with st.expander(f"🖼️ Result {i+1}: {result['filename']}", expanded=i==0):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(result["original_image"], caption=result["filename"], use_container_width=True)
                    
                    # Show detected plate candidates
                    if result["plate_candidates"]:
                        st.markdown("**🎯 Detected Plate Candidates:**")
                        for j, (x, y, w, h, area) in enumerate(result["plate_candidates"]):
                            enhanced_plate = enhance_plate_region(result["phases"]["restored"], x, y, w, h)
                            st.image(enhanced_plate, caption=f"Candidate {j+1} (Area: {area:.0f})", channels="GRAY")
                
                with col2:
                    st.markdown("### 🎯 Recognition Results")
                    
                    # Display results with improved styling
                    plate_status = "✅ Detected" if result['plate_text'] else "❌ Not detected"
                    confidence_color = "#2e8b57" if result['plate_text'] else "#dc3545"
                    
                    info_html = f"""
                    <div style="background: linear-gradient(135deg, #f0f2f6, #e8f4f8); padding: 20px; border-radius: 15px; margin: 10px 0; border-left: 5px solid {confidence_color};">
                        <h4 style="margin: 0 0 15px 0; color: #1f77b4;">🔢 License Plate Recognition</h4>
                        <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <p style="font-size: 24px; font-weight: bold; margin: 0; color: #333; text-align: center;">
                                {result['plate_text'] if result['plate_text'] else 'NO TEXT DETECTED'}
                            </p>
                            <p style="text-align: center; margin: 5px 0 0 0; color: {confidence_color}; font-weight: bold;">
                                {plate_status}
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #e8f5e8, #f0fff0); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #2e8b57;">
                        <h4 style="margin: 0 0 10px 0; color: #2e8b57;">📍 State/Region</h4>
                        <p style="font-size: 18px; font-weight: bold; margin: 0; color: #333;">
                            {result['state']}
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #fff0e6, #fffaf0); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #ff8c00;">
                        <h4 style="margin: 0 0 10px 0; color: #ff8c00;">🚗 Vehicle Classification</h4>
                        <p style="font-size: 18px; font-weight: bold; margin: 0; color: #333;">
                            {result['vehicle_type']}
                        </p>
                    </div>
                    """
                    st.markdown(info_html, unsafe_allow_html=True)
                    
                    # Technical details
                    st.markdown("### 📈 Technical Details")
                    if result["plate_candidates"]:
                        for j, (x, y, w, h, area) in enumerate(result["plate_candidates"]):
                            with st.container():
                                st.markdown(f"""
                                **🎯 Candidate {j+1}:**
                                - **Position:** ({x}, {y})
                                - **Size:** {w} × {h} pixels
                                - **Area:** {area:.0f} pixels²
                                - **Aspect Ratio:** {w/h:.2f}
                                - **Confidence:** {'High' if j == 0 else 'Medium' if j == 1 else 'Low'}
                                """)
                    else:
                        st.warning("⚠️ No license plate candidates detected")
                        st.markdown("""
                        **Possible reasons:**
                        - Image quality too low
                        - License plate not clearly visible
                        - Unusual plate format
                        - Poor lighting conditions
                        """)
    else:
        st.info("👆 Please upload and process images first to view results.")

with tab4:
    st.header("ℹ️ Help & Information")
    
    st.markdown("""
    ## 🚗 Malaysian License Plate Recognition System
    
    This system uses advanced computer vision and OCR techniques to detect and recognize Malaysian license plates.
    
    ### 🎯 How It Works
    
    1. **Image Upload**: Upload clear images containing Malaysian license plates
    2. **9-Phase Processing**: Images go through comprehensive preprocessing
    3. **Plate Detection**: Computer vision algorithms locate potential license plates
    4. **OCR Recognition**: PaddleOCR extracts text from detected plates
    5. **State Identification**: System identifies the issuing state/region
    
    ### 📋 Processing Phases Explained
    
    | Phase | Description | Purpose |
    |-------|-------------|---------|
    | 1️⃣ **Acquisition** | Original image input | Starting point |
    | 2️⃣ **Enhancement** | Histogram equalization + gamma correction | Improve contrast |
    | 3️⃣ **Restoration** | Bilateral filtering | Reduce noise while preserving edges |
    | 4️⃣ **Color Processing** | HSV value channel extraction | Isolate brightness information |
    | 5️⃣ **Wavelet Transform** | Detail coefficient analysis | Highlight edges and textures |
    | 6️⃣ **Compression** | Simulate compression effects | Handle compressed images |
    | 7️⃣ **Morphological** | Gradient operations | Enhance object boundaries |
    | 8️⃣ **Segmentation** | Adaptive thresholding | Create binary image |
    | 9️⃣ **Detection** | Contour analysis | Locate license plate regions |
    
    ### 🏷️ Malaysian License Plate Format
    
    Malaysian license plates follow specific patterns:
    
    - **Standard Format**: [State Code][Number][Letter(s)]
    - **Examples**: 
      - `WJJ1234A` (Kuala Lumpur)
      - `BJH5678` (Selangor)
      - `PGK9999Z` (Penang)
    
    ### 🗺️ State Codes
    
    | Code | State/Territory |
    |------|----------------|
    | A | Perak |
    | B | Selangor |
    | C | Pahang |
    | D | Kelantan |
    | F | Putrajaya |
    | J | Johor |
    | K | Kedah |
    | L | Labuan |
    | M | Melaka |
    | N | Negeri Sembilan |
    | P | Penang |
    | Q | Sarawak |
    | R | Perlis |
    | S | Sabah |
    | T | Terengganu |
    | V/W | Kuala Lumpur |
    | Z | Military |
    
    ### 💡 Tips for Best Results
    
    1. **Image Quality**:
       - Use high-resolution images (minimum 800x600)
       - Ensure good lighting conditions
       - Avoid blurry or motion-blurred images
    
    2. **Plate Visibility**:
       - Capture plates straight-on (avoid angles)
       - Ensure the entire plate is visible
       - Clean plates work better than dirty ones
    
    3. **File Format**:
       - Supported: PNG, JPG, JPEG
       - Maximum size: 10MB per file
       - Multiple files can be processed at once
    
    ### ⚠️ Limitations
    
    - **OCR Accuracy**: Depends on image quality and plate condition
    - **Detection Range**: Works best with standard Malaysian plates
    - **Special Plates**: Custom or diplomatic plates may not be recognized
    - **Processing Time**: Complex images may take longer to process
    
    ### 🔧 Troubleshooting
    
    **No Text Detected?**
    - Check image quality and lighting
    - Ensure plate is clearly visible
    - Try different image angles
    
    **Wrong State Identification?**
    - Verify the detected text is correct
    - Some special series plates may not follow standard patterns
    
    **Slow Processing?**
    - Large images take more time
    - Multiple files are processed sequentially
    - OCR model loading happens once per session
    
    ### 🚀 Running the Application
    
    **Important**: This is a Streamlit application that must be run using:
    
    ```bash
    streamlit run your_script_name.py
    ```
    
    **DO NOT** run with `python your_script_name.py` as this will cause ScriptRunContext errors.
    
    ### 📞 Support
    
    If you encounter issues:
    1. Check the error messages in the interface
    2. Verify your image meets the requirements
    3. Ensure PaddleOCR is properly installed
    4. Check the console for detailed error logs
    """)
    
    # System status check
    st.markdown("---")
    st.subheader("🔍 System Status Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📦 Dependencies:**")
        try:
            import cv2
            st.success("✅ OpenCV installed")
        except ImportError:
            st.error("❌ OpenCV not found")
        
        try:
            import numpy as np
            st.success("✅ NumPy installed")
        except ImportError:
            st.error("❌ NumPy not found")
        
        try:
            from PIL import Image
            st.success("✅ PIL/Pillow installed")
        except ImportError:
            st.error("❌ PIL/Pillow not found")
        
        try:
            import pywt
            st.success("✅ PyWavelets installed")
        except ImportError:
            st.error("❌ PyWavelets not found")
    
    with col2:
        st.markdown("**🔤 OCR Status:**")
        if OCR_ENGINE:
            st.success(f"✅ {OCR_ENGINE.upper()} loaded successfully")
            st.info("🎯 Ready for text recognition")
        else:
            st.error("❌ No OCR engine available")
            st.warning("⚠️ Text recognition will not work")
        
        st.markdown("**💾 Session State:**")
        if "processing_results" in st.session_state:
            result_count = len(st.session_state.processing_results)
            st.info(f"📊 {result_count} results in memory")
        else:
            st.info("🔄 No results stored")

# Footer
st.markdown("---")
footer_html = f"""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 2rem;'>
    <h4 style='color: #495057; margin: 0 0 10px 0;'>🚗 Malaysian License Plate Recognition System</h4>
    <p style='color: #6c757d; margin: 0;'>
        OCR Engine: <strong>{OCR_ENGINE.upper() if OCR_ENGINE else 'None'}</strong> | 
        9-Phase Image Processing Pipeline | 
        Built with Streamlit & OpenCV
    </p>
    <p style='color: #6c757d; margin: 10px 0 0 0; font-size: 0.9em;'>
        💡 Remember to run with: <code>streamlit run your_script.py</code>
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)