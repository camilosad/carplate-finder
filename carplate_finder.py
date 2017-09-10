import cv2
import numpy as np
import sys


def get_approx_contour(contour):
  peri = cv2.arcLength(contour, True)
  return cv2.approxPolyDP(contour, 0.06 * peri, True)


def looks_like_plate(contours):
  if len(contours) != 4:
    return False

  x, y, w, h = cv2.boundingRect(contours)
  return is_large_rectangle(w, h)


def is_large_rectangle(width, height):
  return (width / float(height)) >= 2


def get_largest(contours):
  areas = [cv2.contourArea(c) for c in contours]
  max_index = np.argmax(areas)
  return contours[max_index]


def find_plate_contours(contours):
  potential_contours = []

  for c in contours:
   approx = get_approx_contour(c)
   if looks_like_plate(approx):
    potential_contours.append(approx)

  return get_largest(potential_contours)

def draw_plate_on_image(car_img, plate_contours):
  copy_original_img = original_img.copy()
  cv2.drawContours(copy_original_img, [plate_contours], -1, (0, 255, 0), 3)
  return copy_original_img

def extract_plate_from_image(car_img, plate_contours):
  x, y, w, h = cv2.boundingRect(plate_contours)
  return car_img[y: y + h, x: x + w]


if len(sys.argv) < 2:
  print 'Error: Missing argument'
  print 'Usage: python carplate_finder.py <image_path>'
  exit(0)


image_path = sys.argv[1]

original_img = cv2.imread(image_path)

# Gray scale
gray_img = cv2.cvtColor(original_img,cv2.COLOR_RGB2GRAY)

# Noise removal
noise_free_img = cv2.bilateralFilter(gray_img, 9, 75, 75)

# Threshold
ret, threshold_image = cv2.threshold(noise_free_img, 0, 255, cv2.THRESH_OTSU)

# Canny Edge
canny_image = cv2.Canny(threshold_image, 250, 255)
canny_image = cv2.convertScaleAbs(canny_image)

# Dilatation
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations = 1)

# Find contours
contours, hierarchy = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find plate contours
plate_contours = find_plate_contours(contours)

# Show original image with plate in evidence
# drawn_image = draw_plate_on_image(original_img, plate_contours)
# cv2.imshow("Original image with plate in evidence", drawn_image)

# Show only the plate
plate_img = extract_plate_from_image(original_img, plate_contours)
cv2.imshow("Plate", plate_img)

cv2.waitKey()