import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

img_rgb = iio.imread('gambar.jpg')
img_gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

histogram_counts = np.bincount(img_gray.ravel(), minlength=256)

print("=== Jumlah Total Piksel per Intensitas ===")
for intensitas, jumlah in enumerate(histogram_counts):
    print(f"Intensitas {intensitas:3d}: {jumlah:8d} piksel")

intensitas_dominan = np.argmax(histogram_counts)
jumlah_maksimum = histogram_counts[intensitas_dominan]

print("\n=== Analisis Intensitas Dominan ===")
print(f"Intensitas paling dominan : {intensitas_dominan}")
print(f"Jumlah piksel            : {jumlah_maksimum} piksel")

plt.figure(figsize=(10, 5))
plt.bar(range(256), histogram_counts, color='gray', width=1.0, edgecolor='none')
plt.title('Histogram Gambar Grayscale')
plt.xlabel('Nilai Intensitas Piksel (0 - 255)')
plt.ylabel('Frekuensi / Jumlah Piksel')
plt.xlim([0, 255])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
