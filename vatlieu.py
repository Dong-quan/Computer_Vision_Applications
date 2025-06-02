import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

def vatlieu():
    st.write("Tải lên một hình ảnh và xem liệu nó là **gỗ**, **kim loại** hay **vải**.")

    # Xác định đường dẫn tuyệt đối đến file model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'material_recognition_model.h5')

    # Tải mô hình đã huấn luyện
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        return

    # Danh sách tên lớp – bạn đã biết trước rồi
    class_names = ['go', 'kim_loai', 'vai']  # Cập nhật đúng tên lớp bạn đã dùng khi huấn luyện

    # Chọn file ảnh
    uploaded_file = st.file_uploader("📤 Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc và hiển thị ảnh
        image = Image.open(uploaded_file).resize((150, 150))
        st.image(image, caption="📷 Hình ảnh đã tải lên", use_column_width=True)

        # Tiền xử lý ảnh
        img_array = np.array(image) / 255.0  # chuẩn hóa ảnh
        img_array = np.expand_dims(img_array, axis=0)  # thêm batch dimension

        # Dự đoán
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        # Hiển thị kết quả
        st.success(f"🔎 **Dự đoán:** {predicted_class_name}")
        st.write(f"📈 **Độ tin cậy:** {confidence:.2f}")
