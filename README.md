



# 🎭 Face Detector
![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Transform your face into something extraordinary! A real time face detection app with unique and fun AR mask filters powered by OpenCV and MediaPipe.

![](/assets/casual.jpg)

## ✨ What Makes This Special?

Ever wanted to become a dragon? Or maybe try on a Cyber? Our Face Detector doesnt just detect faces - it transforms them! With our collection of unique AR masks, you can:

![](/assets/cybermask.jpg)

- 🐲 Dragon
- 👑 Venetian
- 😈 Demon
- 🤖 Cyber


All filters work in real time and adapt to your facial movements!


## ✨ Features

- 🔍 Multiple detection methods:
  - Haar Cascade Classifier
  - Deep Neural Network (DNN) detector
- 📊 Configurable confidence threshold
- 🎯 Optional facial landmarks visualization
- 📝 Comprehensive logging system
- ⏱️ Timestamp-based file saving
- 🛠️ Command-line interface for easy usage


A robust and flexible face detection system built with Python and OpenCV, supporting multiple detection methods and visualization options.



3. (Optional) For DNN detection method:
   - Create directory: `models/face_detector/`
   - Download required model files:
     - `res10_300x300_ssd_iter_140000.caffemodel`
     - `deploy.prototxt`
   - Place them in the `models/face_detector/` directory

## 📂 Project Structure
```
face_detection/
├── face_detection.py         # Main script
├── models/                   # Model directory
│   └── face_detector/       # DNN model files
├── results/                 # Output directory
└── README.md
```

### Advanced Options
```bash
# Using DNN method with confidence threshold
python face_detection.py path/to/image.jpg --method dnn --confidence 0.6

# Save output to specific location with landmarks
python face_detection.py path/to/image.jpg --output results/detected.jpg --landmarks

# Full example with all options
python face_detection.py path/to/image.jpg \
    --output results/detected.jpg \
    --method dnn \
    --confidence 0.6 \
    --landmarks
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `image_path` | Path to input image | Required |
| `--output` | Output image path | `detected_faces.jpg` |
| `--method` | Detection method (`haar` or `dnn`) | `haar` |
| `--confidence` | Detection confidence threshold | `0.5` |
| `--landmarks` | Draw facial landmarks | `False` |

## 📋 Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy

## 🎯 Features in Detail

### Detection Methods

1. **Haar Cascade Classifier**
   - Fast and lightweight
   - Good for basic face detection
   - Works well in controlled environments

2. **DNN Detector**
   - More accurate detection
   - Better handling of different face angles
   - Requires additional model files

### Visualization Options

- Rectangle drawing around detected faces
- Optional facial landmarks
- Detection count display
- Configurable output format



## 🚀 Quick Start

### Option 1: Using Anaconda (Recommended)
```bash
# Clone the repository
git clone https://github.com/rexzea/Face-Detector.git

# Navigate Folder
cd Face-Detector

# Install dependencies
conda install -c conda-forge opencv
conda install mediapipe
pip install -r requirements.txt

```

### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/rexzea/Face-Detector.git

# Install dependencies
pip install -r requirements.txt

```

## 🎮 Controls Casual Face Detector

- `q` - Break
- `m` - Show Mesh
- `c` - Show Contours
- `s` - Show Metrics
- `p` - Screenshot

## 🎮 Controls Filter Face Detector
- `q` - Break
- `n` - Change Mask
- `s` - Screenshot
  

## 🛠️ Technical Details

### Core Technologies
- **OpenCV** - For realtime image processing and camera handling
- **MediaPipe** - For precise facial landmark detection
- **Python 3.8+** - Core programming language
- **NumPy** - For efficient numerical operations

### Features
- Realtime face detection and tracking
- unique AR mask filters
- Simple filter animations
- Multi face support
- Photo capture functionality
- Filter customization options

## 🔧 Troubleshooting

Having issues? Here are some common solutions:

1. **Camera not working?**
   - Check your camera specifications

2. **Slow performance?**
   - Make sure youre using GPU enabled environment (if available)
   - Lower the resolution in settings
   - Close other camera applications

3. **Dependencies issues?**
   ```bash
   # Try using Anaconda environment (recommended)
   conda create -n face_detector python=3.8
   conda activate face_detector
   conda install -c conda-forge opencv mediapipe
   ```


## 🤝 Contributing

Got ideas for making this more? Contributions are welcome!

1. 🍴 Fork the repository
2. 🌱 Create your feature branch (`git checkout -b feature/NewFeature`)
3. 💫 Add your changes (`git add .`)
4. 📝 Commit your changes (`git commit -m 'Add some NewFeature'`)
5. 🚀 Push to the branch (`git push origin feature/NewFeature`)
6. 🎉 Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- MediaPipe team for their amazing facial landmark detection
- OpenCV community for computer vision tools
- All our amazing contributors



## 📞 Support & Contact
Need assistance? Reach out through:
- 📧 Email: [futzfary@gmail.com](mailto:futzfary@gmail.com)
- 📱 Phone: +62 898-8610-455
- 💬 GitHub Issues: Open a new issue in the repository

<div align="center">

![Logo Python](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

```
🌟 Crafted with ❤️ by Rexzea 🌟
```
</div>

---

<div align="center">

### Show Your Support
⭐ Star this repository if you find it helpful! ⭐

[Report Bug](https://github.com/rexzea/Face-Detector/issues) · [Request Feature](https://github.com/rexzea/Face-Detector/issues)
