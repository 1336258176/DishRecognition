# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from inference import *
from config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Interactive Interface for Dish-Identification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Interactive Interface for Dish-Identification')
st.markdown(
    """
    Upload pictures to automatically identify the name of the dishes (**currently only Japanese dishes are supported**) 
    """
)

model_path = ""
conf_thres = None
iou_thres = None
model = None
classes = None
with st.sidebar:
    st.header("DL Model Configuration")
    task_type = st.selectbox(
        "Task Type",
        ["Detection"]
    )
    model_type = None
    if task_type == "Detection":
        model_type = st.sidebar.selectbox(
            "Model",
            DETECTION_MODEL_LIST
        )
    else:
        st.error("Currently only 'Detection' function is implemented")

    if model_type:
        model_path = Path(MODEL_DIR, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")

    conf_thres = float(st.slider("Model Confidence", 0, 100, 25)) / 100
    iou_thres = float(st.slider("Model IOU Threshold", 0, 100, 50)) / 100

try:
    model = Inference(model_path)
    classes = getClassesName(MODEL_CONFIG)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

with st.sidebar:
    st.header("Image/Video Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source",
        SOURCES_LIST
    )

source_img = None
if source_selectbox == SOURCES_LIST[0]:  # Video
    source_img = st.file_uploader('Choose an image',
                                  type=['png', 'jpg', 'jpeg'],
                                  help="Support format：png, jpg, jpeg")
    ori_image = None
    image = None
    if source_img:
        file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
        ori_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = ori_image.copy()

    col1, col2 = st.columns(2)
    with col1:
        if ori_image is not None:
            st.image(image=ori_image,
                     caption="Uploaded image",
                     use_container_width=True,
                     channels="BGR")

    if image is not None:
        if st.button("execution"):
            with st.spinner("running..."):
                detection_res = model.inference(image, conf_thres=conf_thres, iou_thres=iou_thres)
                detected_img = draw_img(ori_image.copy(), detection_res, classes)

                with col2:
                    st.image(image=detected_img,
                             caption="Detection Results",
                             use_container_width=True,
                             channels="BGR")
                    try:
                        with st.expander("Detection Results", expanded=True):
                            class_names = []
                            confidences = []
                            for detection_t in detection_res:
                                confidences.append(detection_t[4])
                                class_names.append(classes[int(detection_t[5])])
                            df = pd.DataFrame(
                                {
                                    'class name': class_names,
                                    'confidences': confidences
                                }
                            )
                            st.dataframe(df,
                                         column_config={
                                             'class name': st.column_config.TextColumn(
                                                 '检测到的菜品',
                                                 max_chars=50
                                             ),
                                             'confidences': st.column_config.NumberColumn(
                                                 '置信度',
                                                 help='模型对预测结果的确定性度量',
                                                 min_value=0,
                                                 max_value=1,
                                                 step=0.001,
                                                 format="%.3f"
                                             )
                                         },
                                         hide_index=True,
                                         use_container_width=True)
                    except Exception as e:
                        st.write("No image is uploaded yet!")
                        st.write(e)
else:
    st.error("Currently only 'Image' source are implemented")
