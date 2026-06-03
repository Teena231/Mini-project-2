import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# LEVIR TEST DATASET PATH
# =========================================
base_path = r"C:\Users\ramya\OneDrive\Desktop\building-project\cdtest"

a_path = os.path.join(base_path, "A")
b_path = os.path.join(base_path, "B")
label_path = os.path.join(base_path, "label")

# =========================================
# PICK ONE FILE
# =========================================
file_name = os.listdir(a_path)[106]   # first file automatically

before_img_path = os.path.join(a_path, file_name)
after_img_path = os.path.join(b_path, file_name)
mask_img_path = os.path.join(label_path, file_name)

# =========================================
# READ IMAGES
# =========================================
before = cv2.imread(before_img_path)
after = cv2.imread(after_img_path)
label = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

# check if files loaded properly
if before is None:
    print("Before image not loaded. Check A folder path.")
    exit()

if after is None:
    print("After image not loaded. Check B folder path.")
    exit()

if label is None:
    print("Label image not loaded. Check label folder path.")
    exit()

# convert BGR to RGB for display
before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

# =========================================
# BINARY CHANGE MASK
# =========================================
change_mask = (label > 0).astype(np.uint8)
# =========================================
# REFINED CHANGE USING IMAGE DIFFERENCE
# =========================================
diff = cv2.absdiff(before, after)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

_, thresh = cv2.threshold(gray_diff, 30, 1, cv2.THRESH_BINARY)

# combine with original change mask
refined_change = change_mask & thresh
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("🏗️ Building Change Detection System")

st.write("Upload Before and After Satellite Images")

# Upload images
before_file = st.file_uploader("Upload Before Image (A)", type=["png", "jpg"])
after_file = st.file_uploader("Upload After Image (B)", type=["png", "jpg"])

if before_file and after_file:

    # Read images
    before = cv2.imdecode(np.frombuffer(before_file.read(), np.uint8), 1)
    after = cv2.imdecode(np.frombuffer(after_file.read(), np.uint8), 1)

    before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

    st.subheader("Input Images")

    col1, col2 = st.columns(2)
    col1.image(before, caption="Before Image")
    col2.image(after, caption="After Image")

    # ========================
    # CHANGE DETECTION LOGIC
    # ========================
    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # remove noise
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ========================
    # AREA CALCULATION
    # ========================
    changed_pixels = np.sum(clean > 0)

    meter_per_pixel = 0.5
    area_m2 = changed_pixels * (meter_per_pixel ** 2)
    area_sqft = area_m2 * 10.764

    # ========================
    # OVERLAY
    # ========================
    overlay = after.copy()
    overlay[clean > 0] = [255, 0, 0]

    # ========================
    # DISPLAY OUTPUT
    # ========================
    st.subheader("Results")

    col1, col2 = st.columns(2)
    col1.image(clean, caption="Detected Change Mask")
    col2.image(overlay, caption="Change Overlay")

    st.write("### 📊 Change Analysis")
    st.write(f"Changed Pixels: {changed_pixels}")
    st.write(f"Area: {area_m2:.2f} sq. meters")
    st.write(f"Area: {area_sqft:.2f} sq. feet")

    # ========================
    # ALERT SYSTEM
    # ========================
    if changed_pixels > 500:
        st.error("⚠️ ALERT: Significant building change detected. Authority verification required.")
    else:
        st.info("ℹ️ Minor change detected.")

# =========================================
# AREA CALCULATION
# LEVIR resolution = 0.5 meter/pixel
# =========================================
changed_pixels = np.sum(refined_change)

meter_per_pixel = 0.5
area_m2 = changed_pixels * (meter_per_pixel ** 2)
area_sqft = area_m2 * 10.764

print("File used:", file_name)
print("Changed pixels:", int(changed_pixels))
print("Changed area (sq. meters):", round(area_m2, 2))
print("Changed area (sq. feet):", round(area_sqft, 2))

# =========================================
# ALERT LOGIC
# =========================================
if changed_pixels > 500:
    print("ALERT: Significant building-related change detected. Authority verification required.")
else:
    print("INFO: Minor building-related change detected.")

# =========================================
# OPTIONAL VISUAL OVERLAY
# =========================================
overlay = after.copy()
overlay[refined_change == 1] = [255, 0, 0] # red overlay for changed regions

# =========================================
# VISUALIZATION
# =========================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Before Image (A)")
plt.imshow(before)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("After Image (B)")
plt.imshow(after)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Change Mask (Label)")
plt.imshow(change_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Changed Area Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
