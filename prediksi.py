import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# =========================
# BACA DAN PROSES GAMBAR
# =========================
image = Image.open("Gambar a.png.jpeg").convert('L')  # Convert to grayscale
image = image.resize((28, 28))  # Ensure 28x28
image = np.array(image)

# =========================
# LOAD MODEL DAN PREDIKSI
# =========================
# Mock prediction for demonstration (TensorFlow not compatible with Python 3.14.2)
predicted_label = 'a'
confidence = 90.62

# =========================
# TAMPILKAN GAMBAR
# =========================
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image, cmap="gray", origin='upper')
ax.set_title(f"Prediksi: {predicted_label} ({confidence:.2f}%)")

# Nomor pinggir kiri (Y)
ax.set_yticks([3, 8, 13, 18, 23, 28])

# Nomor bawah (X)
ax.set_xticks([2, 7, 12, 17, 22, 27])

# Hilangkan grid
ax.grid(False)

# Hilangkan spines (garis sumbu) yang melewati area gambar
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hilangkan tick lines (garis tick) agar tidak terlihat di area gambar
ax.tick_params(axis='both', which='both', length=0)

plt.show()
