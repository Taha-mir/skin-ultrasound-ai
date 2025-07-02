import cv2
import numpy as np
import os

def generate_ultrasound_like(shape=(224, 224), abnormal=False):
    img = np.random.normal(128, 25, shape).astype(np.uint8)

    if abnormal:
        for _ in range(np.random.randint(1, 5)):
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            radius = np.random.randint(10, 30)
            cv2.circle(img, center, radius, (255,), -1)  # ضایعه مصنوعی سفید
    return img

os.makedirs('data/processed/normal', exist_ok=True)
os.makedirs('data/processed/abnormal', exist_ok=True)

# تولید داده
for i in range(200):
    normal = generate_ultrasound_like()
    cv2.imwrite(f'data/processed/normal/normal_{i}.png', normal)

    abnormal = generate_ultrasound_like(abnormal=True)
    cv2.imwrite(f'data/processed/abnormal/abnormal_{i}.png', abnormal)

print("✅ داده‌های مصنوعی ساخته شدند.")
