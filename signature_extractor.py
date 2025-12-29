"""Extract signatures from an image."""
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 17th September 2018
# ----------------------------------------------

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np

# the parameters are used to remove small size connected pixels outliar 
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100

# the parameter is used to remove big size connected pixels outliar
constant_parameter_4 = 18

# read the input image
img = cv2.imread('./inputs/cropped.png', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

# connected component analysis by scikit-learn framework
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)

fig, ax = plt.subplots(figsize=(10, 6))

'''
# plot the connected components (for debugging)
ax.imshow(image_label_overlay)
ax.set_axis_off()
plt.tight_layout()
plt.show()
'''

the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0
for region in regionprops(blobs_labels):
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1
    # print region.area # (for debugging)
    # take regions with large enough areas
    if (region.area >= 250):
        if (region.area > the_biggest_component):
            the_biggest_component = region.area

average = (total_area/counter)
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))

# experimental-based ratio calculation, modify it for your cases
# a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
# are smaller than a4_small_size_outliar_constant for A4 size scanned documents
a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

# experimental-based ratio calculation, modify it for your cases
# a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
# are bigger than a4_big_size_outliar_constant for A4 size scanned documents
a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

# remove the connected pixels are smaller than a4_small_size_outliar_constant
pre_version = morphology.remove_small_objects(blobs_labels, int(a4_small_size_outliar_constant))
# remove the connected pixels are bigger than threshold a4_big_size_outliar_constant 
# to get rid of undesired connected pixels such as table headers and etc.
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outliar_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0
# save the the pre-version which is the image is labelled with colors
# as considering connected components
plt.imsave('pre_version.png', pre_version)

# read the pre-version
img = cv2.imread('pre_version.png', 0)
# ensure binary
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# save the final extracted signatures
# cv2.imwrite('./outputs/output.png', img)

# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
# detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
# output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# output[detected_lines > 0] = [0, 0, 255]

# cv2.imwrite('./outputs/output.png', output)

img_inv = 255 - img

# horizontal kernel, long and thin
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
detected_lines = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, horizontal_kernel)

# create a mask to draw detected lines
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
output[detected_lines > 0] = [255, 0, 0]  # draw detected lines in blue

cv2.imwrite('./outputs/detected_lines.png', output)

# remove lines
img_clean = cv2.subtract(img_inv, detected_lines)
# invert back to original: black signatures on white background
img_clean = 255 - img_clean
cv2.imwrite('./outputs/output.png', img_clean)

# ---------------------------------------------
# Remove small noise after line removal
# ---------------------------------------------

# convert cleaned image to binary
_, img_bin = cv2.threshold(img_clean, 127, 255, cv2.THRESH_BINARY)

# invert for skimage (objects should be True)
img_bin_inv = img_bin == 0

# remove small objects (adjust min_size according to your document)
min_size = 100  # minimum pixel area to keep
img_denoised = morphology.remove_small_objects(img_bin_inv, min_size=min_size)

# convert back to uint8 image
img_denoised = (255 - img_denoised.astype(np.uint8) * 255)

# save the final cleaned signature
cv2.imwrite('./outputs/output_cleaned.png', img_denoised)

# convert to binary (if not already)
_, binary = cv2.threshold(img_denoised, 127, 255, cv2.THRESH_BINARY)

# count non-zero pixels (black strokes)
num_signature_pixels = cv2.countNonZero(255 - binary)  # signature strokes are black

# total image pixels
total_pixels = binary.shape[0] * binary.shape[1]

# simple ratio
stroke_ratio = num_signature_pixels / total_pixels

# define a threshold to decide if handwriting exists
threshold = 0.001  # 0.1% of the image pixels
if stroke_ratio > threshold:
    print("Handwriting/signature detected ✅")
else:
    print("No handwriting/signature detected ❌")