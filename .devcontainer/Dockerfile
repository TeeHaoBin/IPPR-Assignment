# .devcontainer/Dockerfile

# Use a base Python image. Choose the version you selected (e.g., 3.11-bookworm or 3.12-bookworm).
FROM python:3.12-slim-bookworm

# Set environment variable for non-interactive apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# libgl1-mesa-glx: Often required by OpenCV for GUI/display functionalities (even if headless).
# tesseract-ocr: The actual Tesseract OCR engine that pytesseract uses.
# libtesseract-dev: Development files for Tesseract, needed if any library tries to compile against it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container to match your VS Code workspace
WORKDIR /workspaces/project

# Copy requirements.txt from the PARENT directory into the WORKDIR
COPY ../requirements.txt .

# Copy ALL other project contents (including your 'src' folder)
# from the parent directory into the WORKDIR
# COPY ../src ./src

# If you moved app.py inside src/ as suggested earlier,
# then you only need to copy the src folder contents.
# A more general approach is to copy all contents from the parent.
# Let's use a robust copy for all project files assuming
# your local structure is: project-root/requirements.txt, project-root/src/, etc.
# We need to copy these into the container's WORKDIR.

# Let's simplify the copy steps to avoid confusion and ensure everything is there.
# If your project root is /Users/TeeHaoBin/Documents/GitHub/IPPR-Assignment
# and Dockerfile is in /Users/TeeHaoBin/Documents/GitHub/IPPR-Assignment/.devcontainer
# and the build context is the .devcontainer folder:

# Copy requirements.txt from the parent directory (your project root)
COPY ../requirements.txt ./requirements.txt

# Copy the entire 'src' folder from the parent directory (your project root)
COPY ../src ./src

# If you have other files at the project root you want to copy (e.g., README.md)
# COPY ../README.md ./README.md

# Install Python dependencies from requirements.txt
# --no-cache-dir: Reduces image size by not storing pip's cache.
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Set the default user to 'vscode' if you want to avoid running as root.
# This requires the 'devcontainers/python' base image or setting up a 'vscode' user manually.
# For simplicity, if using 'python:3.x-slim-bookworm', staying as root is common for beginners.
# USER vscode