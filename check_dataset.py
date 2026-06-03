import os
import tifffile as tiff
import matplotlib.pyplot as plt

train_path = r"C:\Users\ramya\OneDrive\Desktop\building-project\train"

files = os.listdir(train_path)

image_file = [f for f in files if "_image.tif" in f][0]
label_file = image_file.replace("_image.tif", "_label.tif")

img_path = os.path.join(train_path, image_file)
label_path = os.path.join(train_path, label_file)

img = tiff.imread(img_path)
mask = tiff.imread(label_path)

# Fix dimensions
if len(img.shape) == 3:
    img = img[:, :, :3]

if len(mask.shape) > 2:
    mask = mask[:, :, 0]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Satellite Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Building Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()
