import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import torch
import numpy as np
from src.model_torch import CNNModel

# بارگذاری مدل PyTorch
model = CNNModel()
model.load_state_dict(torch.load("models/skin_ultrasound_model.pt", map_location=torch.device("cpu")))
model.eval()

# پردازش تصویر با Pillow (PIL)
def preprocess_image_pil(pil_image):
    img = pil_image.convert("L").resize((224, 224))  # خاکستری + تغییر اندازه
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
    return img_tensor

# رابط کاربری
st.title("🧠 تحلیل سونوگرافی پوست با هوش مصنوعی (PyTorch + PIL)")

uploaded = st.file_uploader("یک تصویر سونوگرافی آپلود کنید:", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="تصویر آپلود شده", use_column_width=True)

    # پردازش و پیش‌بینی
    input_tensor = preprocess_image_pil(image)
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()

    result = "🚨 غیر نرمال" if prob > 0.5 else "✅ نرمال"
    st.markdown(f"### نتیجه مدل: **{result}**")
    st.markdown(f"درصد اطمینان: `{prob:.2f}`")
