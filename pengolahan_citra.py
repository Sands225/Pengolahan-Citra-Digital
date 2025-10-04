import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def read_image(img):
    img_bgr = cv2.imread(img)
    
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_bgr.shape
    print(f'image shape: \nheight: {h} \nwidth: {w} \nchannel: {c}')

    # BGR to RGB
    img = np.zeros_like(img_bgr)
    img[:, :, 0] = img_bgr[:, :, 2]
    img[:, :, 1] = img_bgr[:, :, 1]
    img[:, :, 2] = img_bgr[:, :, 0]

    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()

    return img

# pergeseran citra
def translate_image(img, tx=0, ty=0):
    h, w, c = img.shape
    shift_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            new_i = i + ty
            new_j = j + tx

            if 0 <= new_i < h and 0 <= new_j < w:
                shift_img[new_i, new_j] = img[i, j]
    
    return shift_img

# perbesaran citra
def scale_image(img, sx=1, sy=1):
    h, w, c = img.shape
    new_h = int(h * sy)
    new_w = int(w * sx)

    scaled_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            src_i = int(i / sy)
            src_j = int(j / sx)

            if 0 <= src_i < h and 0 <= src_j < w:
                scaled_img[i, j] = img[src_i, src_j]

    return scaled_img

# pencerminan citra
def horizontal_mirror(img):
    mirrored_img = np.zeros_like(img)
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            mirrored_img[i, j] = img[i, w-j-1]

    return mirrored_img 
    
def vertical_mirror(img):
    mirrored_img = np.zeros_like(img)
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            mirrored_img[i, j] = img[h-i-1, j]
            
    return mirrored_img

def combination_mirror(img):
    mirrored_img = np.zeros_like(img)
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            mirrored_img[i, j] = img[h-i-1, w-j-1]
            
    return mirrored_img

# rotasi citra
def rotate_image(img, angle=0):
    h, w, c = img.shape
    
    rad = np.deg2rad(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    new_h = int(abs(w * sin_a) + abs(h * cos_a))
    new_w = int(abs(w * cos_a) + abs(h * sin_a))

    rotated_image = np.zeros((new_h, new_w, c), dtype=np.uint8)

    cx, cy = w // 2, h // 2
    ncx, ncy = new_w // 2, new_h // 2

    for i in range(new_h):
        for j in range(new_w):
            
            x = (j - ncx)
            y = (i - ncy)

            src_x = int(x * cos_a + y * sin_a + cx)
            src_y = int(-x * sin_a + y * cos_a + cy)

            if 0 <= src_x < w and 0 <= src_y < h:
                rotated_image[i, j] = img[src_y, src_x]

    return rotated_image

# ripple effect
def ripple_image(img, ax=1, ay=1, tx=1, ty=1):
    ripple_img = np.zeros_like(img)
    h, w, c = ripple_img.shape

    for i in range(h):
        for j in range(w):
            new_i = i + int(ay * math.sin(2 * math.pi * j / ty))
            new_j = j + int(ax * math.sin(2 * math.pi * i / tx))

            if 0 <= new_i < h and 0 <= new_j < w:
                ripple_img[new_i, new_j] = img[i, j]

    return ripple_img

# grayscale image
def grayscale_image(img):
    h, w, c = img.shape
    gray_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            r, g, b = img[i, j]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_img[i, j] = gray

    return gray_img

# thresholding image
def thresholding_image(img, threshold=128):
    h, w = img.shape
    binary_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if img[i, j] >= threshold:
                binary_img[i, j] = 1
            else:
                binary_img[i, j] = 0

    return binary_img

# double thresholding image
def double_thresholding_image(img, low_threshold=100, high_threshold=200):
    h, w = img.shape
    double_thresh_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if img[i, j] < low_threshold:
                double_thresh_img[i, j] = 0
            elif low_threshold <= img[i, j] < high_threshold:
                double_thresh_img[i, j] = 128
            else:
                double_thresh_img[i, j] = 255

    return double_thresh_img

#grayscale to m-bit
def grayscale_to_mbit(img, m=4):
    h, w = img.shape
    levels = 2 ** m
    step = 256 // levels

    mbit_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            mbit_img[i, j] = (img[i, j] // step) * step

    return mbit_img

#brightness adjustment
def brightness_adjustment(img, value=0):
    h, w, c = img.shape
    bright_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                new_value = img[i, j, k] + value
                bright_img[i, j, k] = np.clip(new_value, 0, 255)

    return bright_img

#contrast adjustment
def contrast_adjustment(img, factor=1.0):
    h, w, c = img.shape
    contrast_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                new_value = 128 + factor * (img[i, j, k] - 128)
                contrast_img[i, j, k] = np.clip(new_value, 0, 255)

    return contrast_img

# negative image
def negative_image(img):
    h, w, c = img.shape
    neg_img = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            neg_img[i, j] = 255 - neg_img[i, j]

    return neg_img


image = "dola_image.jpg"
img = read_image(image)

translated = translate_image(img, tx=50, ty=30)
scaled = scale_image(img, sx=0.5, sy=0.5)
h_mirror = horizontal_mirror(img)
v_mirror = vertical_mirror(img)
comb_mirror = combination_mirror(img)
rotated = rotate_image(img, angle=45)
rippled = ripple_image(img, ax=10, ay=10, tx=30, ty=30)
gray = grayscale_image(img)
thresh = thresholding_image(gray, threshold=128)
double_thresh = double_thresholding_image(gray, 100, 200)
mbit = grayscale_to_mbit(gray, m=3)
bright = brightness_adjustment(img, value=50)
contrast = contrast_adjustment(img, factor=1.5)
negative = negative_image(img)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

images = [
    (img, "Original"),
    (translated, "Translated"),
    (scaled, "Scaled"),
    (h_mirror, "Horizontal Mirror"),
    (v_mirror, "Vertical Mirror"),
    (comb_mirror, "Combination Mirror"),
    (rotated, "Rotated 45°"),
    (rippled, "Ripple"),
    (gray, "Grayscale"),
    (thresh, "Thresholding"),
    (double_thresh, "Double Threshold"),
    (mbit, "3-bit Grayscale"),
    (bright, "Brightness +50"),
    (contrast, "Contrast x1.5"),
    (negative, "Negative"),
]

for ax, (im, title) in zip(axes.flatten(), images):
    if len(im.shape) == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()