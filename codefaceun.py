import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import random
import os

def khuonmat():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    required_files = {
        'svc': 'svcfaceunknown.pkl',
        'le': 'label_encoderfaceunknown.pkl',
        'detector': 'face_detection_yunet_2023mar.onnx',
        'recognizer': 'face_recognition_sface_2021dec.onnx',
    }

    missing_files = [v for k, v in required_files.items() if not os.path.exists(os.path.join(BASE_DIR, v))]
    if missing_files:
        for file in missing_files:
            st.error(f"❌ Không tìm thấy file: {file}")
        return

    svc = joblib.load(os.path.join(BASE_DIR, required_files['svc']))
    le = joblib.load(os.path.join(BASE_DIR, required_files['le']))
    detector = cv.FaceDetectorYN.create(
        os.path.join(BASE_DIR, required_files['detector']), "", (640, 640), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(
        os.path.join(BASE_DIR, required_files['recognizer']), "")

    threshold = 0.95
    colors_map = {
        'Loc': (255, 0, 0),
        'Nhan': (0, 255, 0),
        'Phong': (0, 0, 255),
        'Loi': (0, 255, 255),
        'Liem': (0, 0, 0)
    }
    used_colors = list(colors_map.values())

    def get_random_color():
        while True:
            color = tuple(random.randint(0, 255) for _ in range(3))
            if color not in used_colors:
                return color

    option = st.radio("Chọn nguồn video", ['📷 Webcam', '📁 Tải video'])

    FRAME_WINDOW = st.image([])

    if option == '📷 Webcam':
        run = st.checkbox("▶ Bắt đầu nhận diện bằng webcam")
        cap = cv.VideoCapture(0)
    else:
        uploaded_file = st.file_uploader("📤 Tải video lên (.mp4, .avi)", type=['mp4', 'avi'])
        if uploaded_file is not None:
            tpath = os.path.join(BASE_DIR, 'temp_video.mp4')
            with open(tpath, 'wb') as f:
                f.write(uploaded_file.read())
            cap = cv.VideoCapture(tpath)
            run = True
        else:
            run = False

    if not run:
        return

    if not cap.isOpened():
        st.error("❌ Không mở được nguồn video.")
        return

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])
    tm = cv.TickMeter()

    while run:
        ret, frame = cap.read()
        if not ret:
            if option == '📁 Tải video':
                st.success("✅ Đã xử lý hết video.")
            else:
                st.warning("⚠ Không lấy được khung hình từ webcam.")
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if faces[1] is not None:
            for face in faces[1]:
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)

                if len(face_feature.shape) == 1:
                    face_feature = face_feature.reshape(1, -1)

                prob = svc.predict_proba(face_feature)[0]
                max_prob = np.max(prob)
                pred_class = np.argmax(prob)

                if max_prob < threshold:
                    name = "Unknown"
                    color = get_random_color()
                else:
                    name = le.inverse_transform([pred_class])[0]
                    color = colors_map.get(name, get_random_color())

                x, y, w, h = map(int, face[:4])
                cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        fps = tm.getFPS()
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        FRAME_WINDOW.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    cap.release()
    cv.destroyAllWindows()

