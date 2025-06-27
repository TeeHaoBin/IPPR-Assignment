from PIL import Image
import cv2
import numpy as np
import streamlit as st
import pywt

st.header("🧪 Image Processing Phases")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="📥 Phase 1: Image Acquisition", use_container_width=True)

    # Phase 2: Image Enhancement (Histogram Equalization)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.equalizeHist(gray)
    st.image(enhanced, caption="✨ Phase 2: Image Enhancement", channels="GRAY")

    # Phase 3: Image Restoration (Gaussian Blur)
    restored = cv2.GaussianBlur(enhanced, (5, 5), 0)
    st.image(restored, caption="🔧 Phase 3: Image Restoration", channels="GRAY")

    # Phase 4: Color Image Processing (HSV)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    st.image(hsv[:, :, 0], caption="🌈 Phase 4: Hue Channel", channels="GRAY")

    # Phase 5: Wavelet Transform (PyWavelets)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    # Normalize LL to [0, 255] and convert to uint8 for display
    LL_norm = np.uint8(255 * (LL - np.min(LL)) / (np.max(LL) - np.min(LL)))
    st.image(LL_norm, caption="🌊 Phase 5: Wavelet LL", channels="GRAY")

    # Phase 6: Image Compression (simulate by resizing)
    compressed = cv2.resize(img_np, (100, 100))
    st.image(compressed, caption="📦 Phase 6: Simulated Compression", use_container_width=True)

    # Phase 7: Morphological Processing (Opening)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    st.image(morph, caption="🧱 Phase 7: Morphological Opening", channels="GRAY")

    # Phase 8: Segmentation (Threshold)
    _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    st.image(segmented, caption="🧩 Phase 8: Segmentation", channels="GRAY")

    # Phase 9: Representation & Description (Contours)
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_np.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    st.image(contour_img, caption="🧬 Phase 9: Contour Representation", use_container_width=True)

    # Optional Phase 10: Object Recognition
    st.markdown("✅ *Phase 10: Object Recognition already handled by OCR module.*")