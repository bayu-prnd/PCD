import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca gambar RGB dan mengonversi ke Grayscale
# Gunakan rumus bobot luminansi: Y = 0.299*R + 0.587*G + 0.114*B
img_rgb = iio.imread('gambar.jpg')
img_gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# 2. Hitung jumlah total piksel untuk setiap nilai intensitas (0-255)
# np.bincount menghitung frekuensi setiap nilai integer dari 0 hingga minlength-1
histogram_counts = np.bincount(img_gray.ravel(), minlength=256)

# a. Menampilkan jumlah total piksel per intensitas (0 - 255)
print("=== Jumlah Total Piksel per Intensitas ===")
for intensitas, jumlah in enumerate(histogram_counts):
    print(f"Intensitas {intensitas:3d}: {jumlah:8d} piksel")

# Menentukan intensitas dominan
intensitas_dominan = np.argmax(histogram_counts)
jumlah_maksimum = histogram_counts[intensitas_dominan]

print("\n=== Analisis Intensitas Dominan ===")
print(f"Intensitas paling dominan : {intensitas_dominan}")
print(f"Jumlah piksel            : {jumlah_maksimum} piksel")

# b. Plot Histogram menggunakan Matplotlib
plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram_counts, color='gray', width=1.0, edgecolor='none')
plt.title('Histogram Gambar Grayscale')
plt.xlabel('Nilai Intensitas Piksel (0 - 255)')
plt.ylabel('Frekuensi / Jumlah Piksel')
plt.xlim([0, 255])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()