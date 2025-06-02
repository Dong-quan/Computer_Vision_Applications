import streamlit as st
import av
import cv2
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from random import choice
import os

def keo():
    # Kiểm tra model tồn tại
    model_path = "rock-paper-scissors-model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model 'rock-paper-scissors-model.h5' không tồn tại.")
        return

    # Load model
    model = load_model(model_path)

    # Class mapping
    REV_CLASS_MAP = {0: "Rock", 1: "Paper", 2: "Scissors", 3: "None"}

    def mapper(val):
        return REV_CLASS_MAP[val]

    def calculate_winner(move1, move2):
        if move1 == move2:
            return "Tie"
        if move1 == "Rock":
            return "User" if move2 == "Scissors" else "Computer"
        if move1 == "Paper":
            return "User" if move2 == "Rock" else "Computer"
        if move1 == "Scissors":
            return "User" if move2 == "Paper" else "Computer"

    # UI
    st.markdown("📸 **Đưa tay vào khung vuông bên trái để chơi.**")

    # Sidebar hướng dẫn
    st.sidebar.markdown("### 📖 Hướng dẫn ký hiệu tay")
    st.sidebar.image("images/Rock.png", caption="Rock ✊", width=100)
    st.sidebar.image("images/Paper.png", caption="Paper ✋", width=100)
    st.sidebar.image("images/Scissors.png", caption="Scissors ✌", width=100)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.prev_move = None
            self.computer_move_name = "None"
            self.winner = "Waiting..."

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            frame_copy = img.copy()

            # Vẽ vùng tay và máy
            cv2.rectangle(frame_copy, (70, 70), (370, 370), (255, 255, 255), 2)
            cv2.rectangle(frame_copy, (420, 70), (620, 270), (255, 255, 255), 2)

            # Dự đoán tay người dùng
            roi = frame_copy[70:370, 70:370]
            img_resized = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (227, 227))

            pred = model.predict(np.array([img_resized]), verbose=0)
            move_code = np.argmax(pred[0])
            user_move_name = mapper(move_code)

            if self.prev_move != user_move_name:
                if user_move_name != "None":
                    self.computer_move_name = choice(["Rock", "Paper", "Scissors"])
                    self.winner = calculate_winner(user_move_name, self.computer_move_name)
                else:
                    self.computer_move_name = "None"
                    self.winner = "Waiting..."
                self.prev_move = user_move_name

            # Hiển thị thông tin
            font = cv2.FONT_HERSHEY_SIMPLEX
            height = frame_copy.shape[0]

            cv2.putText(frame_copy, f"Your Move: {user_move_name}", (100, 50), font, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_copy, f"Computer: {self.computer_move_name}", (380, 50), font, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_copy, f"Winner: {self.winner}", (130, height - 20), font, 1.2, (0, 0, 255), 2)

            # Hiển thị icon của máy tính
            icon_path = f"images/{self.computer_move_name}.png"
            if self.computer_move_name != "None" and os.path.exists(icon_path):
                icon = cv2.imread(icon_path)
                icon = cv2.resize(icon, (200, 200))
                frame_copy[70:270, 420:620] = icon

            return av.VideoFrame.from_ndarray(frame_copy, format="bgr24")

    # Stream camera
    webrtc_streamer(
        key="keo-bua-bao",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
