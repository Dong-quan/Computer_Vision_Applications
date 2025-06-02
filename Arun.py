import streamlit as st
from codefaceun import khuonmat
from nhandangvuongtron import vuongtron
from AAA import AAA
from vatlieu import vatlieu
from play640x480 import keo
from traicay import traicay
from doituong import doituong
st.set_page_config(
    page_title="Ứng dụng Thị Giác Máy",
    page_icon="",
    layout="wide"
)

# ======= Header có logo + tiêu đề =======
col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/logo_ute.png", width=100)  # Đặt logo trường tại thư mục images
with col2:
    st.markdown("""
        <h1 style='color: #4CAF50; margin-bottom:0;'> Ứng dụng Thị Giác Máy</h1>
        <p style='font-size:20px; margin-top:0;'>Chọn chức năng từ thanh bên để bắt đầu</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ======= Sidebar đẹp =======
st.sidebar.title("🔍 Chức năng")
app = st.sidebar.radio(
    "Chọn chương trình:",
    [
        "📚 Bài tập chương 3-4-9",
        "📸 Nhận diện khuôn mặt",
        "⭕ Nhận diện hình vuông – tròn – tam giác",
        "🪵🔍 Nhận dạng vật liệu",
        "🍉Nhận diện trái cây",
        "🧍Nhận diện đối tượng Yolo11n",
        "✊ ✋ ✌ Game Kéo – Búa – Bao Real-time"
        

    ]
)

# ======= Xử lý lựa chọn =======
if app == "📚 Bài tập chương 3-4-9":
    st.header("📚 Bài tập chương 3, 4 và 9")
    AAA()

elif app == "📸 Nhận diện khuôn mặt":
    st.header("📸 Nhận diện khuôn mặt")
    khuonmat()

elif app == "⭕ Nhận diện hình vuông – tròn – tam giác":
    st.header("⭕ Nhận diện hình học: Vuông - Tròn - Tam Giác")
    vuongtron()

elif app == "🪵🔍 Nhận dạng vật liệu":
    st.header("🪵🔍 Nhận dạng vật liệu: Vải - Gỗ - Kim loại")
    vatlieu()

elif app == "🍉Nhận diện trái cây":
    st.header("🍉 Nhận diện & Đếm trái cây")
    traicay()

elif app == "🧍Nhận diện đối tượng Yolo11n":
    st.header("🧍Nhận diện đối tượng Yolo11n")
    doituong()

elif app == "✊ ✋ ✌ Game Kéo – Búa – Bao Real-time":
    st.header("✊ ✋ ✌ Game Kéo – Búa – Bao Real-time")
    keo()

# ======= Footer thông tin nhóm =======
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:16px;'>
    <strong>Hướng dẫn bởi:</strong><br>
    ThS. Trần Tiến Đức<br>        
    <strong>Thực hiện bởi:</strong><br>
    Huỳnh Võ Phúc Lộc – 22146344<br>
    Nguyễn Thiện Nhân – 22146364<br>
    <em>Đại học Sư phạm Kỹ thuật TP. HCM</em>
</div>
""", unsafe_allow_html=True)
