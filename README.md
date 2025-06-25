# 🍽️ Interactive Dish Recognition System

A Streamlit-based web application for automatic Japanese dish identification using YOLOv5 deep learning models.

## 🚀 Features

- **Real-time Detection**: Upload images and get instant dish recognition results
- **Interactive UI**: User-friendly interface with configurable parameters
- **Multiple Models**: Support for different YOLOv5 model variants (yolov5s, yolov5l)
- **Confidence Scoring**: Adjustable confidence and IoU thresholds
- **Detailed Results**: Visual detection results with confidence scores and statistics
- **Performance Optimized**: Model caching for faster inference

## 📋 Supported Dishes

The system can recognize 100 different Japanese dishes including:
- Rice dishes (rice, eels on rice, pilaf, etc.)
- Noodles (udon, soba, ramen, etc.) 
- Sushi and sashimi
- Tempura and fried foods
- Soups (miso soup, chinese soup, etc.)
- Grilled and sauteed dishes
- Desserts and confections

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/1336258176/DishRecognition.git
   cd DishRecognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   Ensure you have the ONNX model files in the `model/` directory:
   - `model/yolov5s.onnx`
   - `model/yolov5l.onnx`

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Configure settings**
   - Select a model (yolov5s or yolov5l) in the sidebar
   - Adjust confidence threshold (default: 0.25)
   - Adjust IoU threshold (default: 0.50)

3. **Upload and analyze**
   - Upload an image (PNG, JPG, JPEG)
   - Click "Execute Detection"
   - View results with bounding boxes and confidence scores

## 📊 Performance Optimizations

### Model Caching
- Uses `@st.cache_resource` to cache loaded models
- Prevents redundant model loading on page refresh
- Significantly improves response time

### Session State Management
- Maintains application state across interactions
- Better user experience with preserved settings

### Error Handling
- Robust exception handling throughout the pipeline
- User-friendly error messages
- Graceful fallbacks for missing configurations

## 🏗️ Architecture

<div align="center">
  <img src="./assets/菜品识别.png" alt="图片描述" width="80">
</div>

```
app.py                 # Main Streamlit application
├── load_model()       # Cached model loading
├── load_classes()     # Cached class name loading
├── render_sidebar()   # UI configuration panel
├── process_image()    # Image inference pipeline
└── display_results()  # Results visualization

inference.py           # Core inference engine
├── Inference class    # ONNX model wrapper
├── preprocess()       # Image preprocessing
├── postprocess()      # NMS and coordinate transformation
└── draw_img()         # Result visualization

config.py             # Configuration management
└── list_files()      # Model discovery utilities
```

## 🔧 Configuration Files

- `model/UECFOOD100/UECFOOD100.yaml`: Class names and model metadata
- `requirements.txt`: Python dependencies
- `config.py`: Application configuration