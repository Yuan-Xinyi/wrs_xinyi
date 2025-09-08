import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt

# 1. Read and binarize image
img_path = "0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/bezier_curve/kanji_2.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"❌ Cannot read image: {img_path}")

_, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV)

# 2. Trace bitmap using potrace
bitmap = potrace.Bitmap(bw)
path = bitmap.trace()

# 3. Prepare three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Subplot 1: Original ---
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original image")
axes[0].axis("equal")
axes[0].axis("off")

# --- Subplot 2: Reconstruction ---
axes[1].set_title("Reconstructed Bézier curves")
axes[1].axis("equal")
axes[1].axis("off")

# --- Subplot 3: Overlay ---
axes[2].imshow(img, cmap="gray")
axes[2].set_title("Overlay (original + curves)")
axes[2].axis("equal")
axes[2].axis("off")

# Draw curves
for curve in path:
    start_point = np.array(curve.start_point)
    last_point = start_point.copy()

    for segment in curve:
        end_point = np.array(segment.end_point)

        if segment.is_corner:
            # Straight corner segment (two line segments)
            c = np.array(segment.c)
            for ax in [axes[1], axes[2]]:
                ax.plot([last_point[0], c[0]], [last_point[1], c[1]], 'r-')
                ax.plot([c[0], end_point[0]], [c[1], end_point[1]], 'r-')
        else:
            # Cubic Bézier curve
            c1 = np.array(segment.c1)
            c2 = np.array(segment.c2)
            t = np.linspace(0, 1, 50)
            B = ((1 - t) ** 3)[:, None] * last_point \
              + (3 * (1 - t) ** 2 * t)[:, None] * c1 \
              + (3 * (1 - t) * t ** 2)[:, None] * c2 \
              + (t ** 3)[:, None] * end_point
            for ax in [axes[1], axes[2]]:
                ax.plot(B[:, 0], B[:, 1], 'b-')

        last_point = end_point  # update

plt.tight_layout()
plt.show()
