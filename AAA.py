import streamlit as st
import cv2
import numpy as np
from PIL import Image
import chapter3 as c3
import chapter4 as c4
import chapter9 as c9
def AAA():
    
    st.title("Ứng dụng Thị Giác Máy với Xử lý ảnh")

    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg", "bmp", "tif", "webp"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        imgin_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        imgin_gray = cv2.cvtColor(imgin_color, cv2.COLOR_BGR2GRAY)

        # Chọn chế độ xử lý ảnh
        mode = st.radio("Chọn chế độ xử lý ảnh:", ["Ảnh màu", "Ảnh xám"])
        img_input = imgin_color if mode == "Ảnh màu" else imgin_gray

        # Chọn chức năng xử lý
        option = st.selectbox("Chọn chức năng xử lý:", [
            "Chapter3 - Negative",
            "Chapter3 - Logarit",
            "Chapter3 - Negative Color",
            "Chapter3 - Power",
            "Chapter3 - Piecewise Line",
            "Chapter3 - Histogram",
            "Chapter3 - Histequal",
            "Chapter3 - Local Hist",
            "Chapter3 - Hist Stat",
            "Chapter3 - Smool Box",
            "Chapter3 - Smool Gauss",
            "Chapter3 - Median Filter",
            "Chapter3 - Sharp",
            "Chapter3 - Gardien",
            "Chapter4 - Spectrum",
            "Chapter4 - Draw Notch Filter",
            "Chapter4 - Remove Moire Simple",
            "Chapter4 - Remove Period Noise",
            "Chapter4 - Draw Period Notch Filter",
            "Chapter9 - Erosion",
            "Chapter9 - Dilation",
            "Chapter9 - Boundary",
            "Chapter9 - Contour",
            "Chapter9 - ConvexHull",
            "Chapter9 - DefectDetect",
            "Chapter9 - HoleFill",
            "Chapter9 - Connected Components",
            "Chapter9 - Remove Small Rice"
        ])

        imgout = None

        # Tùy chọn xử lý theo chức năng
        if option == "Chapter3 - Negative":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Negative(gray)
        elif option == "Chapter3 - Logarit":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Logarit(gray)
        elif option == "Chapter3 - Negative Color":
            imgout = c3.NegativeColor(img_input)
        elif option == "Chapter3 - Power":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Power(gray)
        elif option == "Chapter3 - Piecewise Line":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Piecewiseline(gray)
        elif option == "Chapter3 - Histogram":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Histogram(gray)
        elif option == "Chapter3 - Histequal":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.HistEqual(gray)
        elif option == "Chapter3 - Local Hist":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.LocalHist(gray)
        elif option == "Chapter3 - Hist Stat":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.HistStat(gray)
        elif option == "Chapter3 - Smool Box":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = cv2.boxFilter(gray, ddepth=-1, ksize=(21, 21))
        elif option == "Chapter3 - Smool Gauss":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = cv2.GaussianBlur(gray, (43, 43), 7)
        elif option == "Chapter3 - Median Filter":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = cv2.medianBlur(gray, 5)
        elif option == "Chapter3 - Sharp":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Sharp(gray)
        elif option == "Chapter3 - Gardien":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c3.Gradien(gray)
        elif option == "Chapter4 - Spectrum":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c4.Spectrum(gray)
        elif option == "Chapter4 - Draw Notch Filter":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c4.DrawNotchFilter(gray)
        elif option == "Chapter4 - Remove Moire Simple":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c4.RemoveMoireSimple(gray)
        elif option == "Chapter4 - Remove Period Noise":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c4.RemovePeriodNoise(gray)
        elif option == "Chapter4 - Draw Period Notch Filter":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c4.DrawNotchPeriodFilter(gray)
        elif option == "Chapter9 - Erosion":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.Erosion(gray)
        elif option == "Chapter9 - Dilation":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.Dilation(gray)
        elif option == "Chapter9 - Boundary":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.Boundary(gray)
        elif option == "Chapter9 - Contour":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.Contour(gray)
        elif option == "Chapter9 - ConvexHull":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.ConvexHull(gray)
        elif option == "Chapter9 - DefectDetect":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.DefectDetect(gray)
        elif option == "Chapter9 - HoleFill":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.HoleFill(gray)
        elif option == "Chapter9 - Connected Components":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.ConnectedCompontents(gray)
        elif option == "Chapter9 - Remove Small Rice":
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if mode == "Ảnh màu" else img_input
            imgout = c9.RemoveSmallRice(gray)

        # Hiển thị ảnh gốc và kết quả
        if imgout is not None:
            col1, col2 = st.columns(2)

            with col1:
                if len(img_input.shape) == 3:
                    st.image(img_input, caption="Ảnh Gốc", channels="BGR")
                else:
                    st.image(img_input, caption="Ảnh Gốc", channels="GRAY")

            with col2:
                if len(imgout.shape) == 3:
                    st.image(imgout, caption="Kết quả", channels="BGR")
                else:
                    st.image(imgout, caption="Kết quả", channels="GRAY")
