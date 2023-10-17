import torch
from utils.tool import *
from module.detector import Detector
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np
import time
from yt_dlp import YoutubeDL
import config

# Youtube DLP options
ydl_opts = {"format": "best"}

if torch.cuda.is_available():
    print("Running inference on GPU...")
    device = torch.device("cuda")
else:
    print("GPU not available, running inference on CPU...")
    device = torch.device("cpu")

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (360, int(360 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res_img = cv2.resize(image, (config.INPUT_WIDTH, config.INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0
    
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    infer_time = (end - start) * 1000
    print("Forward time: %f ms" % infer_time)

    # Post-processing of feature maps
    output = handle_preds(preds, device, conf)

    # Load label names
    LABEL_NAMES = []
    with open(config.NAMES, 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    H, W, _ = image.shape
    scale_h, scale_w = H / config.INPUT_HEIGHT, W / config.INPUT_WIDTH

    # Draw predicted boxes
    for box in output[0]:
        box = box.tolist()

        obj_score = box[4]
        category = "null"
        try:
            category = LABEL_NAMES[int(box[5])]
        except:
            pass

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(image, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    st_frame.image(image,
                   caption="Detected Video\nFPS: " + str(round(1000 / infer_time)),
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    print("Loading weights from: %s" % model_path)
    model = Detector(config.CATEGORY_NUM, True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set the model in evaluation mode
    model.eval()
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
            uploaded_image = cv2.imdecode(file_bytes, 1)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res_img = cv2.resize(uploaded_image, (config.INPUT_WIDTH, config.INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                img = res_img.reshape(1, config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)
                img = torch.from_numpy(img.transpose(0, 3, 1, 2))
                img = img.to(device).float() / 255.0
                
                start = time.perf_counter()
                preds = model(img)
                end = time.perf_counter()
                infer_time = (end - start) * 1000
                print("Forward time: %f ms" % infer_time)

                # Post-processing of feature maps
                output = handle_preds(preds, device, conf)

                # Load label names
                LABEL_NAMES = []
                with open(config.NAMES, 'r') as f:
                    for line in f.readlines():
                        LABEL_NAMES.append(line.strip())

                H, W, _ = uploaded_image.shape
                scale_h, scale_w = H / config.INPUT_HEIGHT, W / config.INPUT_WIDTH

                # Draw predicted boxes
                print(output[0])
                for box in output[0]:
                    box = box.tolist()

                    obj_score = box[4]
                    category = "null"
                    try:
                        category = LABEL_NAMES[int(box[5])]
                    except:
                        pass

                    x1, y1 = int(box[0] * W), int(box[1] * H)
                    x2, y2 = int(box[2] * W), int(box[3] * H)

                    cv2.rectangle(uploaded_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(uploaded_image, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
                    cv2.putText(uploaded_image, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

                with col2:
                    st.image(uploaded_image,
                             caption="Detected Image\nFPS: " + str(round(1000 / infer_time)),
                             channels="BGR",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in output[0]:
                                box = box.tolist()
                                category = LABEL_NAMES[int(box[5])]
                                x1, y1 = int(box[0] * W), int(box[1] * H)
                                x2, y2 = int(box[2] * W), int(box[3] * H)
                                output = "Object " + category + " detected at " + str(x1) + ", " + str(y1) + " and " + str(x2) + ", " + str(y2)
                                st.write(output)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def infer_youtube_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if source_youtube:
        try:
            st.video(source_youtube)
        except Exception as e:
                st.error(f"Error loading video: {e}")

    if source_youtube:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(source_youtube, download=False)
                        url = info_dict.get("url", None)
                    vid_cap = cv2.VideoCapture(url)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                         model,
                                                         st_frame,
                                                         image
                                                         )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

