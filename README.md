# Malaysian License Plate Recognition System (MLPRS)

A comprehensive **Image Processing, Computer Vision & Pattern Recognition** system specifically designed for Malaysian license plate detection and recognition. This system implements a sophisticated 9-phase image processing pipeline using classical computer vision techniques combined with modern OCR technology.

## 🚗 Features

- **9-Phase Image Processing Pipeline**: Systematic approach from image acquisition to text recognition
- **Malaysian-Specific Optimization**: Handles standard single-line, two-line (WSL/2956), taxi, and government plates
- **Multi-Algorithm Detection**: Seven parallel detection methods for robust performance
- **Smart Filtering System**: Three-stage validation with color, texture, and template analysis
- **Real-time Processing**: Interactive web interface with live feedback
- **Format Validation**: Comprehensive Malaysian license plate format verification
- **State Code Recognition**: Identifies Malaysian state codes (A-Z mapping)

## 🏗️ System Architecture

### 9-Phase Processing Pipeline
1. **Image Acquisition**: Format validation and preprocessing
2. **Image Enhancement**: Histogram equalization + gamma correction
3. **Image Restoration**: Bilateral filtering for noise reduction
4. **Color Processing**: HSV value channel extraction
5. **Wavelet Transform**: Daubechies-4 detail coefficient extraction
6. **Image Compression**: Compression effect simulation
7. **Morphological Processing**: Gradient operations for boundary enhancement
8. **Image Segmentation**: Adaptive thresholding
9. **Representation & Description**: Multi-method detection and intelligent scoring

### Detection Methods
- Adaptive thresholding for varying illumination
- Dark region detection for black background plates
- Enhanced edge detection with bilateral filtering
- Contrast-based detection for high-contrast regions
- Bright text detection for taxi/bus plates
- Two-line plate detection for stacked formats
- Bus-specific detection for lower image regions

## 📋 Prerequisites

### Required Software
- [**Docker Desktop**](https://www.docker.com/products/docker-desktop/) (recommended) or Docker Engine
- **Git** for cloning the repository
- **Web Browser** (Chrome, Firefox, Safari, Edge)
- [**Streamlit**](https://streamlit.io)

### Alternative Local Setup
If not using Docker:
- **Python 3.12+**
- **pip** package manager

## 🚀 Quick Start with Docker (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/IPPR-Assignment.git
cd IPPR-Assignment
```

### 2. Build and Run with Docker
```bash
# Build the Docker image
docker build -t ippr-system .

# Run the container
docker run -p 8501:8501 ippr-system
```

### 3. Access the Application
Open your web browser and navigate to:
```
http://localhost:8501
```

## 🛠️ Development Setup with DevContainers

### Prerequisites
- **Visual Studio Code**
- **Docker Desktop**
- **Dev Containers extension** for VS Code

### Setup Steps
1. **Clone and Open in VS Code**
   ```bash
   git clone https://github.com/yourusername/IPPR-Assignment.git
   cd IPPR-Assignment
   code .
   ```

2. **Open in DevContainer**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select the option and wait for container to build

3. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

4. **Access via Port Forward**
   - VS Code will automatically forward port 8501
   - Click the notification or go to `http://localhost:8501`

## 💻 Local Installation (Without Docker)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/IPPR-Assignment.git
cd IPPR-Assignment
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv ippr-env

# Activate virtual environment
# On Windows:
ippr-env\Scripts\activate
# On macOS/Linux:
source ippr-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run src/app.py
```

### 5. Access Application
Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
IPPR-Assignment/
├── .devcontainer/
│   ├── devcontainer.json      # VS Code DevContainer configuration
│   └── Dockerfile             # Development container setup
├── src/
│   ├── app.py                 # Main application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Production container setup
├── README.md                  # This file
├── LICENSE                    # Project license
└── Final_Image.zip            # All the images used in testing

```

## 🖥️ Using the Application

### 1. Upload Images
- Navigate to the "📁 Upload & Process" tab
- Upload PNG, JPG, or JPEG files (max 10MB each)
- Supports single or multiple image upload

### 2. View Processing Phases
- Switch to "🔍 Phase Analysis" tab
- See all 9 processing phases applied to your image
- Understand how each phase transforms the image

### 3. Analyze Results
- Check "📊 Results" tab for detection outcomes
- View detected license plate candidates
- See OCR results and confidence scores
- Review format validation results

### 4. Get Help
- Visit "ℹ️ Help" tab for usage instructions
- Learn about supported Malaysian plate formats
- Understand the detection process

## 🔧 Configuration

### Supported Image Formats
- PNG, JPG, JPEG
- Maximum file size: 10MB
- Recommended resolution: 800x600 to 1920x1080

### Malaysian License Plate Formats
- **Standard**: ABC1234, ABC123A
- **Two-line**: ABC/1234, WSL/2956
- **Government**: Special blue background plates
- **Taxi**: Black background with white/yellow text
- **Motorcycle**: Smaller format plates

### State Codes Supported
```
A=Perak, B=Selangor, C=Pahang, D=Kelantan, F=Putrajaya
J=Johor, K=Kedah, L=Labuan, M=Melaka, N=Negeri Sembilan
P=Penang, Q=Sarawak, R=Perlis, S=Sabah, T=Terengganu
V=KL, W=KL
```

## 🐛 Troubleshooting

### Common Issues

**1. Port 8501 already in use**
```bash
# Kill existing Streamlit processes
pkill -f streamlit
# Or use different port
streamlit run src/app.py --server.port 8502
```

**2. Docker build fails**
```bash
# Clean Docker cache
docker system prune -a
# Rebuild image
docker build --no-cache -t ippr-system .
```

**3. PaddleOCR installation issues**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# For macOS
brew install opencv
```

**4. Memory issues with large images**
- Resize images to maximum 1920x1080
- Use JPEG format instead of PNG for smaller file sizes
- Process images one at a time instead of batch processing

### Performance Optimization
- **CPU**: Minimum 4 cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for dependencies

## 📊 Technical Specifications

### Dependencies
- **Streamlit**: Web interface framework
- **OpenCV**: Computer vision operations
- **PaddleOCR**: Text recognition engine
- **PyWavelets**: Wavelet transform operations
- **NumPy**: Numerical computations
- **Pillow**: Image processing utilities

### Performance Metrics
- **Processing Speed**: 8-10 seconds per image
- **Detection Accuracy**: Optimized for Malaysian plates
- **Memory Usage**: ~3000MB during processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- [**Jackson How**](https://github.com/Jackson3925) - *Initial work* 
- [**Chin Pei Fung**](https://github.com/Chowhound0224) - *Initial work* 
- [**Lee Ee-Ern**](https://github.com/ee-ern) - *Initial work*
- [**Tee Hao Bin**](https://github.com/TeeHaoBin) - *Initial work*

## 🙏 Acknowledgments

- **PaddleOCR Team** for the excellent OCR engine
- **OpenCV Community** for computer vision tools
- **Streamlit Team** for the web framework
- [**Dr. Adeline Sneha John Chrisastum**](https://www.linkedin.com/in/dr-adeline-sneha-10068179) for all the guidance and assistance in IPPR module.  

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the help tab in the application

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local privacy laws when processing license plate images.
