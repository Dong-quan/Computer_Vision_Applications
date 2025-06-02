import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import Counter

def doituong():

    # Load mô hình
    model = YOLO('yolo11n.pt', task='detect')

    # Tải ảnh lên
    img_file_buffer = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp", "tif", "webp"])

    if img_file_buffer is not None:
        # Đọc ảnh
        imgin = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
        imgin = cv2.imdecode(imgin, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(imgin, cv2.COLOR_BGR2RGB), caption="📷 Ảnh gốc", use_container_width=True)

        if st.button('🔍 Dự đoán'):
            # Dự đoán
            results = model.predict(imgin, conf=0.6, verbose=False)
            imgout = imgin.copy()
            annotator = Annotator(imgout)
            names = model.names
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.tolist()

            # Đếm
            label_counter = Counter()
            for box, cls, conf in zip(boxes, clss, confs):
                label = names[int(cls)]
                label_counter[label] += 1
                annotator.box_label(box, label=f"{label} {conf:.2f}", txt_color=(255, 0, 0), color=(255, 255, 255))

            # Tab nhận diện và đếm
            tab1, tab2 = st.tabs(["📌 Nhận dạng", "📊 Đếm số lượng"])

            with tab1:
                st.image(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB), caption="🔎 Ảnh đã nhận diện", use_container_width=True)

            with tab2:
                st.subheader("Tổng hợp số lượng:")
                if label_counter:
                    for label, count in label_counter.items():
                        st.write(f"- **{label.capitalize()}**: {count} đối tượng")
                else:
                    st.write("Không phát hiện đối tượng nào.")

            # Nút lưu ảnh
            if st.button('💾 Lưu ảnh'):
                filename = 'predicted_image.jpg'
                cv2.imwrite(filename, imgout)
                st.success(f"Đã lưu ảnh thành {filename}")
                with open(filename, "rb") as file:
                    st.download_button("📥 Tải ảnh xuống", data=file, file_name=filename, mime="image/jpeg", use_container_width=True)
