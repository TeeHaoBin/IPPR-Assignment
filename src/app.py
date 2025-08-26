#!/usr/bin/env python3
"""
Malaysian License Plate Recognition System (IPPR)
==================================================

A complete license plate recognition system implementing the 9-phase academic pipeline:
1. Image Acquisition
2. Image Enhancement  
3. Image Restoration
4. Color Image Processing
5. Wavelets and Multi-Resolution Processing
6. Image Compression
7. Morphological Processing
8. Image Segmentation
9. Representation & Description

Author: Enhanced for 2-line license plate detection
Version: 2.0 with smart filtering and OCR verification
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

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

# ============================================================================
# OCR INITIALIZATION
# ============================================================================

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
            use_angle_cls=True,
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
        return None, None

# ============================================================================
# SMART FILTERING FUNCTIONS
# ============================================================================

def analyze_color_characteristics(roi_color):
    """Analyze color characteristics specific to Malaysian license plates"""
    if roi_color.size == 0 or len(roi_color.shape) != 3:
        return False, 0.0
    
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Malaysian license plates typically have:
        # - White/light backgrounds with dark text
        # - Black backgrounds with white/yellow text (taxis)
        # - Blue backgrounds with white text (government)
        
        # Check for dominant light background (most common)
        light_pixels = np.sum(v > 180)  # Bright pixels
        dark_pixels = np.sum(v < 75)    # Dark pixels
        total_pixels = roi_color.shape[0] * roi_color.shape[1]
        
        light_ratio = light_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # Pattern 1: Light background with dark text (70%+ light background)
        is_light_plate = light_ratio > 0.6 and dark_ratio > 0.05
        
        # Pattern 2: Dark background with light text (60%+ dark background)
        is_dark_plate = dark_ratio > 0.5 and light_ratio > 0.05
        
        # Pattern 3: Moderate contrast (good mix of light and dark)
        is_contrast_plate = 0.2 <= light_ratio <= 0.8 and 0.1 <= dark_ratio <= 0.6
        
        # Calculate color uniformity (plates have relatively uniform backgrounds)
        bg_pixels = roi_color[v > np.median(v)]  # Background pixels
        if len(bg_pixels) > 10:
            bg_std = np.std(bg_pixels)
            color_uniformity = 1.0 / (1.0 + bg_std / 50.0)  # Lower std = higher uniformity
        else:
            color_uniformity = 0.0
        
        is_plate_color = is_light_plate or is_dark_plate or is_contrast_plate
        confidence = color_uniformity * (0.8 if is_light_plate else 0.6 if is_dark_plate else 0.4)
        
        return is_plate_color, confidence
        
    except Exception:
        return False, 0.0

def analyze_texture_characteristics(roi_gray):
    """Analyze texture patterns typical of license plates"""
    if roi_gray.size == 0:
        return False, 0.0
    
    try:
        # License plates have relatively uniform texture with text patterns
        
        # 1. Calculate gradient magnitude for texture analysis
        grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Check for regular text patterns
        # License plates have regular character spacing
        horizontal_profile = np.mean(gradient_magnitude, axis=0)
        
        # Find peaks in horizontal profile (character boundaries)
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(horizontal_profile, height=np.mean(horizontal_profile) * 0.5)
        except ImportError:
            # Fallback peak detection without scipy
            threshold = np.mean(horizontal_profile) * 0.5
            peaks = []
            for i in range(1, len(horizontal_profile) - 1):
                if (horizontal_profile[i] > threshold and 
                    horizontal_profile[i] > horizontal_profile[i-1] and 
                    horizontal_profile[i] > horizontal_profile[i+1]):
                    peaks.append(i)
            peaks = np.array(peaks)
        
        # Regular spacing indicates text characters
        if len(peaks) >= 2:
            spacings = np.diff(peaks)
            spacing_regularity = 1.0 - (np.std(spacings) / np.mean(spacings)) if np.mean(spacings) > 0 else 0.0
            spacing_regularity = max(0.0, min(1.0, spacing_regularity))
        else:
            spacing_regularity = 0.0
        
        # 3. Texture uniformity (backgrounds should be relatively uniform)
        texture_variance = np.var(roi_gray)
        texture_score = 1.0 / (1.0 + abs(texture_variance - 1500) / 1000.0)  # Optimal around 1500
        
        is_plate_texture = spacing_regularity > 0.3 or texture_score > 0.5
        confidence = (spacing_regularity * 0.6 + texture_score * 0.4)
        
        return is_plate_texture, confidence
        
    except Exception:
        return False, 0.0

def match_plate_template(roi_gray):
    """Template matching for Malaysian license plate characteristics"""
    if roi_gray.size == 0:
        return False, 0.0
    
    try:
        # Create simple templates for character patterns
        h, w = roi_gray.shape
        
        # Template 1: Single line text pattern
        template_1line = np.ones((max(20, h//3), max(60, w//2)), dtype=np.uint8) * 255
        template_1line[h//6:h//6+h//6, :] = 0  # Dark text stripe
        
        # Template 2: Two line text pattern  
        template_2line = np.ones((max(30, h//2), max(40, w//3)), dtype=np.uint8) * 255
        template_2line[h//8:h//8+h//8, :] = 0      # Top text line
        template_2line[h//2:h//2+h//8, :] = 0      # Bottom text line
        
        # Resize templates to match ROI size
        template_1line = cv2.resize(template_1line, (min(w, 100), min(h, 40)))
        template_2line = cv2.resize(template_2line, (min(w, 80), min(h, 60)))
        
        # Resize ROI for template matching
        roi_resized = cv2.resize(roi_gray, (template_1line.shape[1], template_1line.shape[0]))
        
        # Match templates
        match_1line = cv2.matchTemplate(roi_resized, template_1line, cv2.TM_CCOEFF_NORMED)
        match_2line = cv2.matchTemplate(roi_resized, template_2line, cv2.TM_CCOEFF_NORMED)
        
        confidence_1line = np.max(match_1line) if match_1line.size > 0 else 0.0
        confidence_2line = np.max(match_2line) if match_2line.size > 0 else 0.0
        
        best_confidence = max(confidence_1line, confidence_2line)
        is_template_match = best_confidence > 0.3
        
        return is_template_match, best_confidence
        
    except Exception:
        return False, 0.0

# ============================================================================
# ENHANCED CANDIDATE DETECTION
# ============================================================================

def find_plate_candidates_from_binary(binary_image: np.ndarray, original_image: np.ndarray = None) -> List[Tuple[int, int, int, int, float]]:
    """Enhanced helper function to find candidates from binary image with smart filtering"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    candidates = []
    img_area = binary_image.shape[0] * binary_image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0 or w == 0:
            continue
            
        # Improved filtering approach with percentage-based sizing
        aspect_ratio = w / float(h)
        area_ratio = (w * h) / img_area
        
        # 1. FIXED: Much more lenient aspect ratio check (includes ALL valid plates)
        valid_single_line = 1.8 <= aspect_ratio <= 6.0    # Standard car plates (EXPANDED)
        valid_two_line = 0.7 <= aspect_ratio <= 3.0       # 2-line plates (EXPANDED)
        valid_square_2line = 0.5 <= aspect_ratio <= 1.5   # Square plates (EXPANDED)
        valid_motorcycle = 1.0 <= aspect_ratio <= 2.8     # Motorcycles (EXPANDED)
        valid_wide_plates = 2.8 <= aspect_ratio <= 4.0    # Wide plates (EXPANDED)
        valid_very_wide = 4.0 <= aspect_ratio <= 7.0      # Very wide plates (NEW)
        valid_aspect = valid_single_line or valid_two_line or valid_square_2line or valid_motorcycle or valid_wide_plates or valid_very_wide
        
        # 2. FIXED: Balanced size check (works for both cars and motorcycles)  
        valid_size = 0.00001 <= area_ratio <= 0.20  # VERY LENIENT for all plate types
        
        # 3. FIXED: More reasonable minimum dimensions (not too restrictive)
        valid_dimensions = w > 12 and h > 4  # REDUCED further to catch more plates
        
        # 4. FIXED: More lenient maximum dimensions 
        max_w = binary_image.shape[1] * 0.8  # Max 80% of image width (INCREASED)
        max_h = binary_image.shape[0] * 0.6  # Max 60% of image height (INCREASED)
        valid_max_dimensions = w <= max_w and h <= max_h
        
        if not (valid_aspect and valid_size and valid_dimensions and valid_max_dimensions):
            continue
            
        # 5. Rectangularity and shape quality checks
        extent = area / (w * h) if (w * h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Check if contour is approximately rectangular
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Approximate contour to check for rectangular shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter for license plate shape characteristics (FIXED: More lenient)
        is_rectangular = len(approx) >= 3 and len(approx) <= 12  # VERY flexible for all plate types
        good_circularity = 0.02 <= circularity <= 0.98  # VERY RELAXED for all detections
        good_extent = extent > 0.20  # VERY RELAXED for all plate types
        good_solidity = solidity > 0.4   # VERY RELAXED for all conditions
        
        if not (good_extent and good_solidity and is_rectangular and good_circularity):
            continue
            
        # 6. SMART FILTERING: Progressive multi-stage validation
        confidence_score = 0.0
        
        if original_image is not None:
            # Extract regions for analysis
            roi_gray = binary_image[y:y+h, x:x+w] if len(binary_image.shape) == 2 else cv2.cvtColor(binary_image[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            roi_color = original_image[y:y+h, x:x+w] if len(original_image.shape) == 3 else cv2.cvtColor(original_image[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)
            
            # Stage 1: Color Analysis
            is_color_match, color_confidence = analyze_color_characteristics(roi_color)
            confidence_score += color_confidence * 0.3
            
            # Stage 2: Texture Analysis  
            is_texture_match, texture_confidence = analyze_texture_characteristics(roi_gray)
            confidence_score += texture_confidence * 0.3
            
            # Stage 3: Template Matching
            is_template_match, template_confidence = match_plate_template(roi_gray)
            confidence_score += template_confidence * 0.4
            
            # Progressive filtering: Only reject if ALL methods fail
            total_matches = sum([is_color_match, is_texture_match, is_template_match])
            
            # Reject only if confidence is very low AND no method matches
            if confidence_score < 0.1 and total_matches == 0:
                continue  # Reject obvious non-plates
            
            # Boost area score based on confidence
            area = area * (1.0 + confidence_score)  # Confidence boost
        
        candidates.append((x, y, w, h, area))
    
    return candidates

# ============================================================================
# LICENSE PLATE DETECTION METHODS
# ============================================================================

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
        candidates1 = find_plate_candidates_from_binary(adaptive_thresh, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "adaptive") for c in candidates1])
    except:
        pass
    
    # Method 2: Dark region detection (license plates are typically dark)
    try:
        # Invert image to find dark regions as white
        inverted = cv2.bitwise_not(image)
        _, dark_thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2 = find_plate_candidates_from_binary(dark_thresh, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "dark") for c in candidates2])
    except:
        pass
    
    # Method 2b: Standard Otsu thresholding
    try:
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates2b = find_plate_candidates_from_binary(otsu_thresh, image)
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
        
        candidates3 = find_plate_candidates_from_binary(morph, image)
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
        
        candidates4 = find_plate_candidates_from_binary(dark_mask, image)
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
        
        candidates5 = find_plate_candidates_from_binary(bright_mask, image)
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
        bus_candidates = find_plate_candidates_from_binary(bus_processed, lower_third)
        
        # Adjust y-coordinates back to full image coordinates
        adjusted_bus_candidates = []
        for x, y, w, h, area in bus_candidates:
            adjusted_y = y + height//3  # Add offset for lower third
            adjusted_bus_candidates.append((x, adjusted_y, w, h, area, "bus"))
        
        all_candidates.extend(adjusted_bus_candidates)
    except:
        pass
    
    # Method 7: Enhanced two-line license plate detection (motorcycles)
    try:
        # Apply adaptive threshold optimized for 2-line plates
        adaptive_2line = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
        # MOTORCYCLE-SPECIFIC: Use vertical morphological operations to connect stacked text
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))  # Tall kernel for vertical connection
        morph_2line = cv2.morphologyEx(adaptive_2line, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Also apply horizontal connection within each line
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))  # Wide kernel for horizontal connection
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # Additional morphological operations to merge the two lines
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 12))  # Medium width, taller for 2-line merging
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_CLOSE, kernel_merge)
        
        # Clean up noise while preserving main structures
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        morph_2line = cv2.morphologyEx(morph_2line, cv2.MORPH_OPEN, kernel_clean)
        
        candidates_2line = find_plate_candidates_from_binary(morph_2line, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "2line") for c in candidates_2line])
    except:
        pass
    
    # Method 8: ENHANCED MOTORCYCLE-SPECIFIC detection (small, square plates)
    try:
        # Use smaller, more aggressive thresholding for small motorcycle plates
        motorcycle_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        
        # STAGE 1: Micro-motorcycles (very small distant plates)
        # Use tiny morphological kernels for very small text
        kernel_micro_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Tiny horizontal connection
        kernel_micro_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))  # Tiny vertical connection
        
        micro_processed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, kernel_micro_h)
        micro_processed = cv2.morphologyEx(micro_processed, cv2.MORPH_CLOSE, kernel_micro_v)
        
        # Very light final connection
        kernel_micro_final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        micro_processed = cv2.morphologyEx(micro_processed, cv2.MORPH_CLOSE, kernel_micro_final)
        
        candidates_micro = find_plate_candidates_from_binary(micro_processed, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "micro_motorcycle") for c in candidates_micro])
        
        # STAGE 2: Regular motorcycles (standard size)
        # Use slightly larger kernels for normal motorcycle plates
        kernel_moto_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # Small horizontal connection
        kernel_moto_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # Small vertical connection
        
        moto_processed = cv2.morphologyEx(motorcycle_thresh, cv2.MORPH_CLOSE, kernel_moto_h)
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_CLOSE, kernel_moto_v)
        
        # Final connection with small kernel
        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_CLOSE, kernel_final)
        
        # Very light cleanup to preserve small plates
        kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        moto_processed = cv2.morphologyEx(moto_processed, cv2.MORPH_OPEN, kernel_cleanup)
        
        candidates_motorcycle = find_plate_candidates_from_binary(moto_processed, image)
        all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], "motorcycle") for c in candidates_motorcycle])
        
    except:
        pass
    
    # Method 9: MULTI-SCALE MOTORCYCLE detection (handles various distances)
    try:
        # Create multiple scales to handle plates at different distances
        h_img, w_img = image.shape[:2]
        
        # Scale 1: Original size for close motorcycles
        scale1_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        
        # Use progressive kernels for different text sizes
        for kernel_size in [(2, 1), (3, 2), (4, 3)]:  # Very small kernels
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            processed = cv2.morphologyEx(scale1_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Add vertical connection for 2-line plates
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size[1] + 2))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_v)
            
            candidates_scale = find_plate_candidates_from_binary(processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"multiscale_{kernel_size[0]}x{kernel_size[1]}") for c in candidates_scale])
            
        # Scale 2: Handle very small distant plates with upscaling
        if min(h_img, w_img) > 200:  # Only if image is large enough
            # Create a focused region for distant plate detection (lower 2/3 of image)
            roi_y_start = h_img // 3
            roi = image[roi_y_start:, :]
            
            # Apply more aggressive thresholding for distant small plates
            distant_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)
            
            # Use the smallest possible kernels for distant plates
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            distant_processed = cv2.morphologyEx(distant_thresh, cv2.MORPH_CLOSE, kernel_tiny)
            
            # Add minimal vertical connection
            kernel_tiny_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            distant_processed = cv2.morphologyEx(distant_processed, cv2.MORPH_CLOSE, kernel_tiny_v)
            
            # Find candidates in ROI and adjust coordinates
            roi_candidates = find_plate_candidates_from_binary(distant_processed, roi)
            for x, y, w, h, area in roi_candidates:
                adjusted_y = y + roi_y_start  # Adjust for ROI offset
                all_candidates.append((x, adjusted_y, w, h, area, "distant_motorcycle"))
                
    except:
        pass
    
    # Method 10: EXTREME CASES detection (for very challenging motorcycle plates)
    try:
        
        # CASE 1: Ultra-small distant plates (like KAA17)
        # Use minimal thresholding to catch faint distant plates
        _, ultra_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ultra-minimal morphological operations
        kernel_ultra = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Minimal kernel
        ultra_processed = cv2.morphologyEx(ultra_thresh, cv2.MORPH_CLOSE, kernel_ultra)
        
        # Try to connect very small text pieces with tiny kernels
        for tiny_size in [(2, 1), (1, 2), (3, 1), (1, 3)]:
            tiny_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tiny_size)
            temp_processed = cv2.morphologyEx(ultra_processed, cv2.MORPH_CLOSE, tiny_kernel)
            
            candidates_ultra = find_plate_candidates_from_binary(temp_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"ultra_tiny_{tiny_size[0]}x{tiny_size[1]}") for c in candidates_ultra])
        
        # CASE 2: Dark/low-light plates (like GT41)
        # Enhanced histogram equalization for dark images
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced_dark = clahe.apply(image)
        
        # Apply multiple thresholding approaches for dark images
        dark_methods = [
            (cv2.THRESH_BINARY, "dark_binary"),
            (cv2.THRESH_BINARY_INV, "dark_binary_inv"), 
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "dark_adaptive_gauss"),
            (cv2.ADAPTIVE_THRESH_MEAN_C, "dark_adaptive_mean")
        ]
        
        for thresh_type, method_name in dark_methods:
            try:
                if thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
                    _, dark_thresh = cv2.threshold(enhanced_dark, 0, 255, thresh_type + cv2.THRESH_OTSU)
                else:
                    dark_thresh = cv2.adaptiveThreshold(enhanced_dark, 255, thresh_type, cv2.THRESH_BINARY, 3, 1)
                
                # Very light morphology for dark images
                kernel_dark = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dark_processed = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel_dark)
                
                candidates_dark = find_plate_candidates_from_binary(dark_processed, image)
                all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], method_name) for c in candidates_dark])
            except:
                continue
        
        # CASE 3: Heavily occluded plates (like MBQ73)
        # Use very permissive shape filtering
        # Apply gentle gaussian blur to connect fragmented text
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        _, occluded_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try different dilation strategies to connect broken text
        for dilation_size in [(1, 2), (2, 1), (2, 2), (3, 2), (2, 3)]:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_size)
            occluded_processed = cv2.dilate(occluded_thresh, kernel_dilate, iterations=1)
            
            # Very light erosion to clean up
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            occluded_processed = cv2.erode(occluded_processed, kernel_erode, iterations=1)
            
            candidates_occluded = find_plate_candidates_from_binary(occluded_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"occluded_{dilation_size[0]}x{dilation_size[1]}") for c in candidates_occluded])
        
        # CASE 4: Two-line plates with rider interference (like BNW76)
        # Focus on vertical text connection with rider masking
        
        # Create a mask to reduce rider interference (focus on lower portion)
        h_img, w_img = image.shape
        mask = np.zeros_like(image)
        # Focus on bottom 60% where license plates typically are, avoiding rider torso
        mask[int(h_img * 0.4):, int(w_img * 0.1):int(w_img * 0.9)] = 255
        
        masked_image = cv2.bitwise_and(image, mask)
        
        # Apply aggressive adaptive thresholding on masked region
        masked_thresh = cv2.adaptiveThreshold(masked_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 2)
        
        # Specialized 2-line connection with various vertical kernels
        for v_height in range(3, 12, 2):  # Try different vertical connection heights
            kernel_2line = cv2.getStructuringElement(cv2.MORPH_RECT, (2, v_height))
            two_line_processed = cv2.morphologyEx(masked_thresh, cv2.MORPH_CLOSE, kernel_2line)
            
            # Add horizontal connection within each line
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
            two_line_processed = cv2.morphologyEx(two_line_processed, cv2.MORPH_CLOSE, kernel_h)
            
            candidates_2line_special = find_plate_candidates_from_binary(two_line_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"special_2line_v{v_height}") for c in candidates_2line_special])
        
        # CASE 5: Indoor/artificial lighting plates (like WCQ7)
        # Enhanced gamma correction for indoor lighting
        for gamma in [0.5, 0.7, 1.3, 1.5]:  # Different gamma values
            gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
            
            # Apply strong adaptive thresholding
            indoor_thresh = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1)
            
            # Minimal morphology to preserve small text
            kernel_indoor = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            indoor_processed = cv2.morphologyEx(indoor_thresh, cv2.MORPH_CLOSE, kernel_indoor)
            
            # Add tiny vertical connection for 2-line indoor plates
            kernel_indoor_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            indoor_processed = cv2.morphologyEx(indoor_processed, cv2.MORPH_CLOSE, kernel_indoor_v)
            
            candidates_indoor = find_plate_candidates_from_binary(indoor_processed, image)
            all_candidates.extend([(c[0], c[1], c[2], c[3], c[4], f"indoor_gamma_{gamma}") for c in candidates_indoor])
            
    except Exception as e:
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

# ============================================================================
# FORMAT VALIDATION AND OCR VERIFICATION
# ============================================================================

def validate_malaysian_plate_format(text: str) -> Tuple[bool, float, str]:
    """Validate if text matches Malaysian license plate formats"""
    if not text or len(text) < 3:
        return False, 0.0, "too_short"
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Malaysian plate patterns - CORRECTED based on actual rules
    patterns = [
        # Standard format with suffix: A1A, ABC1234A, WCC9831A (start with alphabet(s), 1-4 integers, optional 1 alphabet)
        (r'^[A-Z]+[0-9]{1,4}[A-Z]$', 0.95, "standard_with_suffix"),
        # Standard format: A1, ABC1234, WCC9831, BLR83 (start with alphabet(s), 1-4 integers)
        (r'^[A-Z]+[0-9]{1,4}$', 0.90, "standard"),
    ]
    
    for pattern, confidence, format_type in patterns:
        if re.match(pattern, clean_text):
            # Additional validation for common Malaysian prefixes
            state_codes = ['A', 'B', 'C', 'D', 'F', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Z']
            
            if format_type in ["standard", "two_line"] and clean_text[0] in state_codes:
                confidence += 0.1  # Bonus for valid state code
            
            return True, confidence, format_type
    
    return False, 0.0, "invalid"

def detect_text_lines(roi_gray):
    """Detect horizontal text lines in the ROI for 2-line processing"""
    if roi_gray.size == 0:
        return []
    
    try:
        # Create horizontal projection
        horizontal_projection = np.sum(roi_gray < 128, axis=1)  # Sum dark pixels horizontally
        
        # Smooth the projection
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(horizontal_projection, sigma=1.0)
        except ImportError:
            # Simple smoothing fallback
            kernel = np.array([0.25, 0.5, 0.25])
            smoothed = np.convolve(horizontal_projection, kernel, mode='same')
        
        # Find peaks (text lines)
        try:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(smoothed, height=np.max(smoothed) * 0.3, distance=len(smoothed)//4)
        except ImportError:
            # Fallback peak detection
            threshold = np.max(smoothed) * 0.3
            min_distance = len(smoothed) // 4
            peaks = []
            
            for i in range(min_distance, len(smoothed) - min_distance):
                if smoothed[i] > threshold:
                    # Check if it's a local maximum
                    is_peak = True
                    for j in range(max(0, i - min_distance), min(len(smoothed), i + min_distance)):
                        if smoothed[j] > smoothed[i]:
                            is_peak = False
                            break
                    if is_peak:
                        peaks.append(i)
            peaks = np.array(peaks)
        
        # Extract line regions
        lines = []
        for peak in peaks:
            # Find line boundaries
            start = peak
            end = peak
            
            # Expand backwards
            while start > 0 and smoothed[start] > np.max(smoothed) * 0.1:
                start -= 1
            
            # Expand forwards  
            while end < len(smoothed) - 1 and smoothed[end] > np.max(smoothed) * 0.1:
                end += 1
            
            if end - start > 5:  # Minimum line height
                lines.append((start, end))
        
        return lines
        
    except Exception:
        return []

def extract_license_plate_text_correct(ocr_model, image: np.ndarray) -> Optional[str]:
    """
    Enhanced OCR function for text extraction from license plates with 2-line support
    
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
                for i, detection in enumerate(results[0], 1):
                    if len(detection) >= 2:
                        text, confidence = detection[1]
                        texts_and_scores.append((text, confidence))
                
                if texts_and_scores:
                    # Sort by confidence to get reliable texts first
                    texts_and_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Try to find single complete plate text first
                    for text, score in texts_and_scores:
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                        if len(clean_text) >= 4 and score > 0.6:  # LOWERED: More lenient for single plates
                            # VALIDATE FORMAT before accepting single plate
                            is_valid, format_conf, format_type = validate_malaysian_plate_format(clean_text)
                            
                            if is_valid:
                                return clean_text
                    
                    # ENHANCED 2-LINE PROCESSING: Better line detection and combination
                    if len(texts_and_scores) >= 2:
                        # First, try to detect text lines using image analysis
                        if len(image.shape) == 3:
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = image
                        
                        text_lines = detect_text_lines(gray)
                        
                        # Get text detections along with their bounding boxes
                        bbox_texts = []
                        for i, detection in enumerate(results[0][:len(texts_and_scores)]):
                            if len(detection) >= 2:
                                bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                text, confidence = detection[1]
                                clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                                
                                if len(clean_text) >= 1 and confidence > 0.2:  # LOWERED: Even more lenient for 2-line parts
                                    # Calculate center coordinates
                                    center_y = sum([point[1] for point in bbox]) / 4
                                    center_x = sum([point[0] for point in bbox]) / 4
                                    
                                    # Calculate text area for weighting
                                    width = max([point[0] for point in bbox]) - min([point[0] for point in bbox])
                                    height = max([point[1] for point in bbox]) - min([point[1] for point in bbox])
                                    area = width * height
                                    
                                    bbox_texts.append((clean_text, confidence, center_y, center_x, area))
                        
                        if len(bbox_texts) >= 2:
                            # Sort by vertical position (top to bottom)
                            bbox_texts.sort(key=lambda x: x[2])
                            
                            # Try different combination strategies
                            best_combination = None
                            best_confidence = 0
                            
                            # Strategy 1: Simple top-bottom combination
                            for i in range(len(bbox_texts)-1):
                                for j in range(i+1, min(len(bbox_texts), i+3)):  # Try next 2 candidates
                                    text1, conf1, y1, x1, area1 = bbox_texts[i]
                                    text2, conf2, y2, x2, area2 = bbox_texts[j]
                                    
                                    # Check if texts are vertically aligned (2-line format)
                                    vertical_distance = abs(y2 - y1)
                                    horizontal_overlap = min(abs(x1 - x2), gray.shape[1] * 0.3)
                                    
                                    if vertical_distance > 5 and horizontal_overlap < gray.shape[1] * 0.5:
                                        # Determine order based on position and typical Malaysian formats
                                        if y1 < y2:  # text1 is above text2
                                            combined = text1 + text2
                                        else:  # text2 is above text1
                                            combined = text2 + text1
                                        
                                        # Validate combination
                                        is_valid, format_conf, _ = validate_malaysian_plate_format(combined)
                                        
                                        if is_valid:
                                            combo_confidence = (conf1 + conf2) / 2 * format_conf
                                            if combo_confidence > best_confidence:
                                                best_combination = combined
                                                best_confidence = combo_confidence
                            
                            # Strategy 2: Fallback - simple concatenation of top 2
                            if not best_combination and len(bbox_texts) >= 2:
                                combined_text = bbox_texts[0][0] + bbox_texts[1][0]
                                avg_confidence = (bbox_texts[0][1] + bbox_texts[1][1]) / 2
                                
                                if len(combined_text) >= 3 and avg_confidence > 0.3:  # LOWERED: More lenient fallback
                                    best_combination = combined_text
                                    best_confidence = avg_confidence
                            
                            if best_combination and best_confidence > 0.3:  # LOWERED: Accept more combinations
                                return best_combination
                    
                    # Fallback: use best single detection
                    best_text = None
                    highest_score = 0
                    
                    for text, score in texts_and_scores:
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                        if len(clean_text) >= 2 and score > highest_score:
                            best_text = clean_text
                            highest_score = score
                    
                    if best_text and len(best_text) >= 3:
                        return best_text
        
        return None
            
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

def ocr_verification_pipeline(ocr_model, ocr_engine: str, candidates: List, phases_dict: Dict) -> List[Tuple]:
    """Run OCR verification on top candidates using multiple phases for better results"""
    if not candidates or ocr_model is None:
        return [(c[0], c[1], c[2], c[3], c[4], "", 0.0) for c in candidates]
    
    verified_candidates = []
    
    # Define OCR phases to try (in order of preference)
    ocr_phases = [
        ("restored", "Phase 3: Bilateral filtered"),
        ("enhanced", "Phase 2: Histogram equalized"), 
        ("color_processed", "Phase 4: HSV Value channel")
    ]
    
    # Process top candidates (expanded to match UI display)
    for i, (x, y, w, h, area) in enumerate(candidates[:10]):
        try:
            best_text = ""
            best_confidence = 0.0
            best_phase = ""
            all_attempts = []
            
            # Try OCR on multiple phases
            for phase_key, phase_name in ocr_phases:
                if phase_key not in phases_dict:
                    continue
                    
                try:
                    # Extract ROI from this phase
                    phase_image = phases_dict[phase_key]
                    enhanced_roi = enhance_plate_region(phase_image, x, y, w, h)
                    
                    # Run OCR on this phase
                    extracted_text = extract_plate_text(enhanced_roi, ocr_model, ocr_engine)
                    
                    if extracted_text:
                        # Validate format
                        is_valid, format_confidence, format_type = validate_malaysian_plate_format(extracted_text)
                        
                        # Calculate confidence for this attempt
                        text_length_score = min(1.0, len(extracted_text) / 6.0)
                        attempt_confidence = text_length_score * 0.5 + format_confidence * 0.5
                        
                        # Bonus for valid Malaysian plate format
                        if is_valid:
                            attempt_confidence *= 1.5
                        
                        all_attempts.append({
                            'text': extracted_text,
                            'confidence': attempt_confidence,
                            'phase': phase_name,
                            'is_valid': is_valid,
                            'format_type': format_type
                        })
                        
                        
                        # Update best result if this is better
                        if attempt_confidence > best_confidence:
                            best_text = extracted_text
                            best_confidence = attempt_confidence
                            best_phase = phase_name
                            
                except Exception as phase_error:
                    continue
            
            # MAJORITY VOTING: When scores are similar, use majority vote
            if all_attempts:
                attempts_str = [(a['text'], f"{a['confidence']:.3f}") for a in all_attempts]
                
                # Group attempts by confidence range (within 0.1 of each other)
                confidence_groups = {}
                for attempt in all_attempts:
                    conf_key = round(attempt['confidence'], 1)  # Round to nearest 0.1
                    if conf_key not in confidence_groups:
                        confidence_groups[conf_key] = []
                    confidence_groups[conf_key].append(attempt)
                
                # Find the highest confidence group
                max_conf_key = max(confidence_groups.keys()) if confidence_groups else 0
                top_group = confidence_groups.get(max_conf_key, [])
                
                # Apply majority voting within the top confidence group
                if len(top_group) > 1:
                    # Count occurrences of each text result
                    text_votes = {}
                    for attempt in top_group:
                        text = attempt['text']
                        if text not in text_votes:
                            text_votes[text] = {'count': 0, 'total_conf': 0, 'attempts': []}
                        text_votes[text]['count'] += 1
                        text_votes[text]['total_conf'] += attempt['confidence']
                        text_votes[text]['attempts'].append(attempt)
                    
                    # Find majority winner
                    max_votes = max(text_votes[text]['count'] for text in text_votes)
                    majority_candidates = [text for text in text_votes if text_votes[text]['count'] == max_votes]
                    
                    if len(majority_candidates) == 1:
                        # Clear majority winner
                        majority_text = majority_candidates[0]
                        majority_data = text_votes[majority_text]
                        best_text = majority_text
                        best_confidence = majority_data['total_conf'] / majority_data['count']
                        best_phase = majority_data['attempts'][0]['phase']
                    
                    elif len(majority_candidates) > 1:
                        # Tie-breaker: prefer longer text (more complete detection)
                        longest_text = max(majority_candidates, key=len)
                        majority_data = text_votes[longest_text]
                        best_text = longest_text
                        best_confidence = majority_data['total_conf'] / majority_data['count']
                        best_phase = majority_data['attempts'][0]['phase']
                
            else:
                pass
            
            # Calculate final OCR confidence
            ocr_confidence = best_confidence
            
            # Penalty for obviously wrong text
            if best_text and (len(best_text) > 10 or any(char in best_text for char in ['@', '#', '$', '%'])):
                ocr_confidence *= 0.1
            
            # Boost area score based on OCR success
            boosted_area = area
            is_valid = any(a['is_valid'] for a in all_attempts)
            
            if is_valid and best_text:
                # MASSIVE boost to ensure valid plates become Candidate 1
                boosted_area *= (1.0 + ocr_confidence * 50.0)  # INCREASED from 3.0 to 50.0
            elif best_text and len(best_text) >= 3:
                boosted_area *= (1.0 + ocr_confidence * 5.0)  # INCREASED from 1.0 to 5.0
            else:
                boosted_area *= 0.1  # INCREASED penalty to push non-readable candidates down
            
            verified_candidates.append((x, y, w, h, boosted_area, best_text, ocr_confidence))
            
        except Exception as e:
            verified_candidates.append((x, y, w, h, area * 0.3, "", 0.0))  # Penalty for OCR failure
    
    # GUARANTEE: Valid license plates always come first
    valid_plates = []
    other_candidates = []
    
    for candidate in verified_candidates:
        x, y, w, h, boosted_area, text, confidence = candidate
        
        # Check if this candidate has a valid Malaysian plate text
        if text:
            is_valid, _, _ = validate_malaysian_plate_format(text)
            if is_valid:
                valid_plates.append(candidate)
            else:
                other_candidates.append(candidate)
        else:
            other_candidates.append(candidate)
    
    # Sort each group by boosted area score
    valid_plates.sort(key=lambda x: x[4], reverse=True)
    other_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Combine: valid plates first, then others
    final_candidates = valid_plates + other_candidates
    
    return final_candidates

# ============================================================================
# IMAGE ENHANCEMENT AND PROCESSING
# ============================================================================

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
        # Return proportional fallback based on aspect ratio
        aspect_ratio = w / h if h > 0 else 2.0
        fallback_h = 50
        fallback_w = int(fallback_h * aspect_ratio)
        return np.zeros((fallback_h, fallback_w), dtype=np.uint8)
    
    # Start with a copy of the ROI
    enhanced_roi = roi.copy()
    
    # SMART enhancement: Different approaches based on region size
    try:
        region_area = enhanced_roi.shape[0] * enhanced_roi.shape[1]
        
        # MOTORCYCLE-SPECIFIC: For small regions (likely motorcycles), use different approach
        if region_area < 3000:  # Small motorcycle plate region
            
            # For small regions, upscale first to help OCR
            scale_factor = max(2, int(60 / min(enhanced_roi.shape[:2])))  # Scale to at least 60px height
            if scale_factor > 1:
                enhanced_roi = cv2.resize(enhanced_roi, 
                                        (enhanced_roi.shape[1] * scale_factor, 
                                         enhanced_roi.shape[0] * scale_factor), 
                                        interpolation=cv2.INTER_CUBIC)
            
            # More aggressive contrast enhancement for small text
            min_val, max_val = np.min(enhanced_roi), np.max(enhanced_roi)
            contrast_range = max_val - min_val
            
            if contrast_range < 150 and contrast_range > 0:  # More aggressive threshold for small plates
                enhanced_roi = ((enhanced_roi - min_val) * 255.0 / contrast_range).astype(np.uint8)
            
            # Light denoising for upscaled small images
            enhanced_roi = cv2.medianBlur(enhanced_roi, 3)
            
        else:  # Larger regions (likely cars) - use gentle enhancement
            
            # Method 1: Simple histogram stretching (very gentle)
            min_val, max_val = np.min(enhanced_roi), np.max(enhanced_roi)
            contrast_range = max_val - min_val
            
            # Only enhance if contrast is very poor (< 100 out of 255)
            if contrast_range < 100 and contrast_range > 0:
                enhanced_roi = ((enhanced_roi - min_val) * 255.0 / contrast_range).astype(np.uint8)
                pass
            else:
                pass
            
            # Method 2: Very light sharpening only for larger regions
            if enhanced_roi.shape[0] > 30 and enhanced_roi.shape[1] > 60:
                # Gentle unsharp mask
                gaussian = cv2.GaussianBlur(enhanced_roi, (3, 3), 1.0)
                enhanced_roi = cv2.addWeighted(enhanced_roi, 1.2, gaussian, -0.2, 0)
        
    except Exception as e:
        pass
        enhanced_roi = roi.copy()
    
    return enhanced_roi

# ============================================================================
# 9-PHASE ACADEMIC PIPELINE
# ============================================================================

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
        
        # Smart edge penalty (less harsh for small motorcycle plates)
        edge_penalty = 1.0
        
        # Check if this is likely a small motorcycle plate (area-based detection)
        is_small_plate = area < 1000  # Likely motorcycle if area < 1000
        
        if (y <= 2 or x <= 2 or  # Only penalize VERY close to edges
            (y + h) >= (img_height - 2) or (x + w) >= (img_width - 2)):  # Near opposite edges
            edge_penalty = 0.001  # Extreme penalty for image border detections
        elif not is_small_plate and (y < 20 or x < 20 or  # Regular edge penalty only for large plates
            (y + h) > (img_height - 20) or (x + w) > (img_width - 20)):  # Regular far edges
            edge_penalty = 0.1  # Moderate penalty for large plates near edges
        elif is_small_plate and (y < 10 or x < 10 or  # More lenient for small plates
            (y + h) > (img_height - 10) or (x + w) > (img_width - 10)):  # More lenient for small plates
            edge_penalty = 0.5  # Light penalty for small plates near edges
        
        # Prefer license plates in lower 2/3 of image (where cars are)
        position_score = 1.0
        center_y = y + h/2
        if center_y > img_height * 0.3:  # Lower 70% of image
            position_score = 2.0  # Strong bonus for car area
        if center_y > img_height * 0.7:  # Bottom 30% of image
            position_score = 1.5  # Still good but slightly lower
        
        # MOTORCYCLE-OPTIMIZED size scoring (prioritizes small plates)
        size_score = 1.0
        if area > 25000:  # Very large areas are likely false positives
            size_score = 0.2
        elif area > 15000:  # Large areas are suspicious
            size_score = 0.4
        elif 5000 <= area <= 15000:  # Good size range for car license plates
            size_score = 1.5
        elif 2000 <= area <= 5000:  # Smaller plates (cars and large motorcycles)
            size_score = 1.8  # Higher bonus for smaller realistic plates
        elif 800 <= area <= 2000:  # MOTORCYCLE SIZE RANGE - very high priority
            size_score = 2.5   # INCREASED bonus for motorcycle-sized plates
        elif 400 <= area <= 800:   # Small motorcycles - very high priority
            size_score = 2.8   # HIGHEST bonus for small plates
        elif 200 <= area <= 400:   # Tiny distant motorcycles - high priority
            size_score = 2.2   # HIGH bonus for tiny plates
        elif 100 <= area <= 200:   # Micro distant motorcycles - still valid
            size_score = 1.8   # Good bonus for micro plates
        elif 50 <= area <= 100:    # Very tiny but possible distant plates
            size_score = 1.2   # Modest bonus
        else:
            size_score = 0.3   # Very small or very large
        
        # IMPROVED aspect ratio preference with motorcycle support
        aspect_ratio = w / h
        aspect_score = 1.0
        if 3.0 <= aspect_ratio <= 4.5:  # Ideal single-line car plates
            aspect_score = 1.2
        elif 1.4 <= aspect_ratio <= 2.2:  # MOTORCYCLE ASPECT RATIO - high priority
            aspect_score = 1.5  # INCREASED bonus for motorcycle aspect ratios
        elif 1.0 <= aspect_ratio <= 1.4:  # Square 2-line plates
            aspect_score = 1.3  # Good bonus for square 2-line plates
        elif 2.2 <= aspect_ratio <= 3.0:  # Between motorcycle and car
            aspect_score = 1.1
        elif 0.7 <= aspect_ratio <= 1.0:  # Very square 2-line plates
            aspect_score = 1.2  # Increased for very square motorcycle plates
        elif 2.0 <= aspect_ratio <= 5.0:  # Acceptable range
            aspect_score = 1.0
        else:
            aspect_score = 0.8
        
        # Method-specific bonuses with EXTREME MOTORCYCLE PRIORITY
        method_score = 1.0
        
        # EXTREME CASES - HIGHEST PRIORITY
        if method.startswith("ultra_tiny_"):  # ULTRA HIGH: Ultra-tiny distant plate detection
            method_score = 15.0
        elif method.startswith("indoor_gamma_"):  # ULTRA HIGH: Indoor lighting detection
            method_score = 12.0
        elif method.startswith("special_2line_"):  # ULTRA HIGH: Special 2-line with rider masking
            method_score = 11.0
        elif method.startswith("occluded_"):  # VERY HIGH: Occluded plate detection
            method_score = 10.0
        elif method.startswith("dark_"):  # VERY HIGH: Dark lighting detection
            method_score = 9.5
        
        # STANDARD MOTORCYCLE DETECTION
        elif method == "micro_motorcycle":  # HIGH: Micro motorcycle detection for distant plates
            method_score = 8.0
        elif method == "distant_motorcycle":  # HIGH: Distant motorcycle detection
            method_score = 7.0
        elif method.startswith("multiscale_"):  # HIGH: Multi-scale detection methods
            method_score = 6.0
        elif method == "motorcycle":  # HIGH: Standard motorcycle-specific detection
            method_score = 5.0
        elif method == "2line":  # HIGH: Two-line plate detection  
            method_score = 4.5
        
        # OTHER METHODS
        elif method == "bus":  # MEDIUM-HIGH: Bus-specific detection
            method_score = 3.0
        elif method == "bright":  # MEDIUM: Light text detection for buses
            method_score = 2.5
        elif method in ["dark", "contrast"]:  # MEDIUM: Good for regular license plates
            method_score = 2.0
        elif method == "adaptive":  # MEDIUM-LOW: Adaptive thresholding
            method_score = 1.8
        elif method == "edge":  # MEDIUM-LOW: Edge detection
            method_score = 1.6
        
        # Additional scoring bonus for license plate-like characteristics
        characteristic_bonus = 1.0
        
        # Bonus for typical license plate positioning (lower part of image, not edges)
        center_x = x + w/2
        center_y = y + h/2
        
        # Smart position-based characteristic bonus (motorcycle-aware)
        if (x <= 5 or y <= 5 or 
            (x + w) >= (img_width - 5) or (y + h) >= (img_height - 5)):
            characteristic_bonus = 0.01  # Extreme penalty for true edge detections
        
        # Enhanced bonus system for different plate types
        elif is_small_plate:  # Different rules for small motorcycle plates
            # Motorcycles can be anywhere in lower 2/3 of image
            if (img_height * 0.2 <= center_y <= img_height * 0.95 and  # Lower 75% of image
                img_width * 0.05 <= center_x <= img_width * 0.95):     # Almost entire width (motorcycles can be at sides)
                characteristic_bonus = 2.0  # Higher bonus for small plates in good positions
        else:  # Rules for larger car plates
            # Cars typically in lower 60% and more centered
            if (img_height * 0.3 <= center_y <= img_height * 0.9 and  # Lower 60% of image
                img_width * 0.1 <= center_x <= img_width * 0.9):       # Central 80% horizontally
                characteristic_bonus = 1.5
        
        # Enhanced size-based bonus for image coverage
        image_coverage = area / (img_width * img_height)
        if is_small_plate:
            # Motorcycle plates: 0.01% to 1% of image area
            if 0.0001 <= image_coverage <= 0.01:  
                characteristic_bonus *= 1.8  # Higher bonus for small plate coverage
            elif 0.01 <= image_coverage <= 0.02:  # Slightly larger motorcycles
                characteristic_bonus *= 1.5
        else:
            # Car plates: 0.2% to 3% of image area  
            if 0.002 <= image_coverage <= 0.03:
                characteristic_bonus *= 1.3
        
        return area_score * edge_penalty * position_score * size_score * aspect_score * method_score * characteristic_bonus
    
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

# ============================================================================
# TEXT EXTRACTION AND STATE IDENTIFICATION
# ============================================================================

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

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error(f"File {uploaded_file.name} is too large (max 10MB)")
        return False
    
    return True

# ============================================================================
# STREAMLIT USER INTERFACE
# ============================================================================

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
                        
                        # ENHANCED: Use OCR verification pipeline for better results
                        plate_text = ""
                        best_candidate = None
                        
                        if plate_candidates and ocr_model:
                            # Run OCR verification on top candidates using multiple phases
                            verified_candidates = ocr_verification_pipeline(ocr_model, ocr_engine, plate_candidates, phases)
                            
                            if verified_candidates:
                                # Get the best verified candidate
                                best_candidate = verified_candidates[0]
                                x, y, w, h, area, extracted_text, ocr_confidence = best_candidate
                                
                                # Use extracted text if available, otherwise fallback to basic OCR
                                if extracted_text:
                                    plate_text = extracted_text
                                else:
                                    # Fallback: basic OCR on best geometric candidate
                                    plate_roi = enhance_plate_region(phases['restored'], x, y, w, h)
                                    plate_text = extract_plate_text(plate_roi, ocr_model, ocr_engine)
                            else:
                                # Fallback: basic OCR on first candidate
                                x, y, w, h, _ = plate_candidates[0]
                                plate_roi = enhance_plate_region(phases['restored'], x, y, w, h)
                                plate_text = extract_plate_text(plate_roi, ocr_model, ocr_engine)
                        
                        # Identify state and vehicle type
                        state = identify_state(plate_text)
                        
                        # Store results
                        result_data = {
                            "filename": uploaded_file.name,
                            "phases": phases,
                            "plate_candidates": plate_candidates,
                            "plate_text": plate_text,
                            "state": state,
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
                ("grayscale", "📥 Phase 1: Pre-Processing", "Gray Scale Image"),
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
    streamlit run IPPR.py
    ```
    
    **DO NOT** run with `python IPPR.py` as this will cause ScriptRunContext errors.
    
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
        💡 Remember to run with: <code>streamlit run IPPR.py</code>
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)