# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from inference import Inference, draw_img
from config import MODEL_DIR, SOURCES_LIST, list_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Interactive Interface for Dish-Identification",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DETECTION_MODEL_LIST = list_files('./model')
DEFAULT_CLASSES = [
    "rice", "eels on rice", "pilaf", "chicken-'n'-egg on rice", "pork cutlet on rice",
    "beef curry", "sushi", "chicken rice", "fried rice", "tempura bowl",
    "bibimbap", "toast", "croissant", "roll bread", "raisin bread",
    "chip butty", "hamburger", "pizza", "sandwiches", "udon noodle",
    "tempura udon", "soba noodle", "ramen noodle", "beef noodle",
    "tensin noodle", "fried noodle", "spaghetti", "Japanese-style pancake",
    "takoyaki", "grilled eggplant", "sauteed vegetables", "croquette",
    "grilled corn", "sauteed spinach", "vegetable tempura", "miso soup",
    "potage", "sausage", "oden", "omelet", "ganmodoki", "jiaozi",
    "stew", "teriyaki grilled fish", "fried fish", "grilled salmon",
    "salmon meuniere", "sashimi", "grilled pacific saury", "sukiyaki",
    "sweet and sour pork", "lemon chicken", "hamburger steak", "steak",
    "dried fish", "ginger pork saute", "spicy chili-flavored tofu",
    "yakitori", "cabbage roll", "rolled omelet", "egg sunny-side up",
    "fermented soybeans", "cold tofu", "egg roll", "chilled noodle",
    "stir-fried beef and peppers", "simmered pork", "boiled fish",
    "seasoned beef with potatoes", "hambarg steak", "beef steak",
    "grilled chicken", "sauteed chicken", "chicken nugget", "fried chicken",
    "chicken teriyaki", "chicken katsu", "roast chicken", "steamed egg hotchpotch",
    "omelet with fried rice", "cutlet curry", "spaghetti meat sauce",
    "fried shrimp", "potato salad", "green salad", "macaroni salad",
    "Japanese tofu and vegetable chowder", "pork miso soup", "chinese soup",
    "beef bowl", "kinpira-style sauteed burdock", "rice ball", "pizza toast",
    "dipping noodles", "hot dog", "french fries", "mixed rice", "goya chanpuru",
    "green curry", "okinawan suteki", "mango pudding", "almond jelly",
    "jelly", "chocolate cake", "cheese cake", "apple pie", "ice cream",
    "shaved ice", "dango", "japanese confection", "chestnut rice", "Japanese pilaf"
]

# Cached functions for performance optimization
@st.cache_resource
def load_model(model_path: str) -> Optional[Inference]:
    """
    Load and cache the inference model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Inference model or None if loading fails
    """
    try:
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            return None
            
        model = Inference(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Unable to load model: {e}")
        return None

@st.cache_data
def load_classes() -> List[str]:
    """
    Load class names from configuration or use default.
    
    Returns:
        List of class names
    """
    try:
        # Try to load from config file if it exists
        config_path = MODEL_DIR / "UECFOOD100" / "UECFOOD100.yaml"
        if config_path.exists():
            from inference import getClassesName
            return getClassesName(str(config_path))
        else:
            logger.warning("Config file not found, using default classes")
            return DEFAULT_CLASSES
    except Exception as e:
        logger.warning(f"Failed to load classes from config: {e}, using default")
        return DEFAULT_CLASSES

def render_sidebar() -> Tuple[Optional[str], float, float]:
    """
    Render the sidebar with model configuration options.
    
    Returns:
        Tuple of (model_path, confidence_threshold, iou_threshold)
    """
    with st.sidebar:
        st.header("🔧 DL Model Configuration")
        
        # Task type selection
        task_type = st.selectbox(
            "Task Type",
            ["Detection"],
            help="Currently only Detection is supported"
        )
        
        # Model selection
        model_type = None
        if task_type == "Detection":
            if not DETECTION_MODEL_LIST:
                st.error("No models found in the model directory")
                return None, 0.25, 0.50
                
            model_type = st.selectbox(
                "Model",
                DETECTION_MODEL_LIST,
                help="Select a YOLOv5 model for detection"
            )
        else:
            st.error("Currently only 'Detection' function is implemented")
            return None, 0.25, 0.50
        
        # Model path
        model_path = None
        if model_type:
            model_path = str(Path(MODEL_DIR, model_type))
        else:
            st.error("Please select a model")
            return None, 0.25, 0.50
        
        # Threshold settings
        st.subheader("Detection Thresholds")
        conf_thres = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25,
            step=0.01,
            help="Minimum confidence score for detections"
        )
        
        iou_thres = st.slider(
            "IoU Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.50,
            step=0.01,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        return model_path, conf_thres, iou_thres

def render_image_upload() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Render image upload interface.
    
    Returns:
        Tuple of (original_image, processed_image)
    """
    with st.sidebar:
        st.header("📁 Image/Video Config")
        source_selectbox = st.selectbox(
            "Select Source",
            SOURCES_LIST,
            help="Choose input source type"
        )
    
    if source_selectbox != SOURCES_LIST[0]:  # Not Image
        st.error("Currently only 'Image' source is implemented")
        return None, None
    
    # File uploader
    source_img = st.file_uploader(
        'Choose an image',
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if source_img is None:
        return None, None
    
    try:
        # Read and decode image
        file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
        ori_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if ori_image is None:
            st.error("Failed to decode image. Please check the file format.")
            return None, None
            
        image = ori_image.copy()
        return ori_image, image
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def process_image(model: Inference, image: np.ndarray, conf_thres: float, iou_thres: float) -> Optional[np.ndarray]:
    """
    Process image with the model and return detection results.
    
    Args:
        model: Inference model
        image: Input image
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        
    Returns:
        Detection results or None if processing fails
    """
    try:
        with st.spinner("🔍 Processing image..."):
            detection_res = model.inference(image, conf_thres=conf_thres, iou_thres=iou_thres)
            return detection_res
    except Exception as e:
        st.error(f"Error during inference: {e}")
        logger.error(f"Inference error: {e}")
        return None

def display_results(original_image: np.ndarray, detection_results: np.ndarray, classes: List[str], col2):
    """
    Display detection results with visualization and data table.
    
    Args:
        original_image: Original input image
        detection_results: Detection results from model
        classes: List of class names
        col2: Streamlit column for display
    """
    try:
        # Draw detection results
        detected_img = draw_img(original_image.copy(), detection_results, classes)
        
        with col2:
            st.image(
                image=detected_img,
                caption="🎯 Detection Results",
                use_container_width=True,
                channels="BGR"
            )
            
            # Display results table
            if len(detection_results) > 0:
                with st.expander("📊 Detection Details", expanded=True):
                    class_names = []
                    confidences = []
                    
                    for detection in detection_results:
                        confidences.append(float(detection[4]))
                        class_names.append(classes[int(detection[5])])
                    
                    df = pd.DataFrame({
                        'Dish Name': class_names,
                        'Confidence': confidences
                    })
                    
                    # Sort by confidence
                    df = df.sort_values('Confidence', ascending=False).reset_index(drop=True)
                    
                    st.dataframe(
                        df,
                        column_config={
                            'Dish Name': st.column_config.TextColumn(
                                '🍽️ Detected Dish',
                                help='Name of the detected dish',
                                max_chars=50
                            ),
                            'Confidence': st.column_config.NumberColumn(
                                '📊 Confidence Score',
                                help='Model confidence in the prediction',
                                min_value=0.0,
                                max_value=1.0,
                                step=0.001,
                                format="%.3f"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.subheader("📈 Summary")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Total Detections", len(detection_results))
                    with col_b:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
                    with col_c:
                        st.metric("Max Confidence", f"{np.max(confidences):.3f}")
            else:
                st.info("No dishes detected. Try adjusting the confidence threshold.")
                
    except Exception as e:
        st.error(f"Error displaying results: {e}")
        logger.error(f"Display error: {e}")

def main():
    """Main application function."""
    # App header
    st.title('🍽️ Interactive Interface for Dish-Identification')
    st.markdown(
        """
        Upload pictures to automatically identify the name of the dishes 
        (**currently only Japanese dishes are supported**) 
        
        📌 **Instructions:**
        1. Configure model settings in the sidebar
        2. Upload an image using the file uploader
        3. Click 'Execute Detection' to analyze the image
        """
    )
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Render sidebar
    model_path, conf_thres, iou_thres = render_sidebar()
    
    # Load model and classes
    model = None
    classes = load_classes()
    
    if model_path:
        model = load_model(model_path)
    
    # Image upload and processing
    original_image, image = render_image_upload()
    
    # Main content layout
    col1, col2 = st.columns(2)
    
    # Display original image
    with col1:
        if original_image is not None:
            st.image(
                image=original_image,
                caption="📸 Uploaded Image",
                use_container_width=True,
                channels="BGR"
            )
    
    # Process button and results
    if image is not None and model is not None:
        if st.button("🚀 Execute Detection", type="primary"):
            detection_results = process_image(model, image, conf_thres, iou_thres)
            
            if detection_results is not None:
                display_results(original_image, detection_results, classes, col2)
                st.session_state.processed = True
    
    elif image is not None and model is None:
        st.error("❌ Model not loaded. Please check the model configuration.")
    elif model is not None and image is None:
        st.info("📤 Please upload an image to start detection.")
    else:
        st.info("🎯 Configure the model and upload an image to begin.")

if __name__ == "__main__":
    main()
