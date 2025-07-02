import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from src.model_torch import CNNModel

# بارگذاری مدل آموزش‌دیده
model = CNNModel()
model.load_state_dict(torch.load("models/skin_ultrasound_model.pt", map_location=torch.device('cpu')))
model.eval()

# تابع پردازش تصویر
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, 1))  # (1, 1, 224, 224)
    return torch.tensor(img)

# رابط کاربری
st.title("🧠 تحلیل سونوگرافی پوست با هوش مصنوعی ")
uploaded = st.file_uploader("یک تصویر سونوگرافی آپلود کنید:", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    temp_path = "temp_input.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    st.image(temp_path, caption="تصویر آپلود شده", use_column_width=True)

    # پردازش و پیش‌بینی
    input_tensor = preprocess_image(temp_path)
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()

    result = "🚨 غیر نرمال" if prob > 0.5 else "✅ نرمال"
    st.markdown(f"### نتیجه مدل: **{result}**")
    st.markdown(f"درصد اطمینان: `{prob:.2f}`")

    os.remove(temp_path)
