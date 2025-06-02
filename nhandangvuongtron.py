import streamlit as st
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
from PIL import Image
import io
import os


def vuongtron():

    # Định nghĩa BASE_DIR là thư mục hiện tại của file này
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    uploaded_file = st.file_uploader("Tải ảnh lên", type=['jpg', 'png', 'jpeg', 'bmp', 'tif', 'webp'])

    if uploaded_file is not None:
        # Đọc file một lần duy nhất
        file_data = uploaded_file.read()

        # Chuyển thành ảnh PIL
        image = Image.open(io.BytesIO(file_data)).convert("RGB")

        # Chuyển thành ảnh OpenCV
        file_bytes = np.asarray(bytearray(file_data), dtype=np.uint8)
        imgin_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        imgin_gray = cv2.cvtColor(imgin_color, cv2.COLOR_BGR2GRAY)

        st.subheader("Ảnh đầu vào:")
        st.image(image, channels="RGB", use_column_width=True)

        if st.button("🔺 Nhận dạng hình học"):
            ketqua = shape_predict(imgin_gray)
            imgout_shape = imgin_color.copy()
            cv2.putText(imgout_shape, ketqua, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            st.image(cv2.cvtColor(imgout_shape, cv2.COLOR_BGR2RGB), caption="Kết quả nhận dạng hình", use_column_width=True)


def phan_nguong(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            s = 255 if r == 63 else 0
            imgout[x, y] = np.uint8(s)
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

def shape_predict(imgin):
    temp = phan_nguong(imgin)
    m = cv2.moments(temp)
    Hu = cv2.HuMoments(m)
    if 0.000624 <= Hu[0, 0] <= 0.000626:
        return 'Hinh Tron'
    elif 0.000644 <= Hu[0, 0] <= 0.000668:
        return 'Hinh Vuong'
    elif 0.000725 <= Hu[0, 0] <= 0.000751:
        return 'Hinh Tam Giac'
    else:
        return 'Không xác định'

def yolo_predict(img, model):
    imgout = img.copy()
    annotator = Annotator(imgout)
    results = model.predict(img, conf=0.6, verbose=False)
    names = model.names

    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.tolist()

    for box, cls, conf in zip(boxes, clss, confs):
        label = f"{names[int(cls)]} {conf:.2f}"
        annotator.box_label(box, label=label, txt_color=(255, 0, 0), color=(255, 255, 255))

    return imgout
