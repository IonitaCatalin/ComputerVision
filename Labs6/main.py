import math 
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

# img should be a h x w x 1 image after canny edge detector
# the function will return the hough response and samples on the most probable line
def compute_hough_space(img, threshold):
  h, w = img.shape
  theta_max = 1.0 * math.pi
  theta_min = 0.0

  rho_min = 0.0
  rho_max = math.hypot(h, w)

  rho_bin = 200
  theta_bin = 300
  hough_space = np.zeros((rho_bin, theta_bin))

  for y in range(h):
    for x in range(w):
      # skip points that are not on the line
      if img[y, x] == 0:
        continue
      else:
        for itheta in range(theta_bin):
          theta = (1.0 * itheta) * (theta_max / theta_bin) # find theta bin
          rho = x * math.cos(theta) + y * math.sin(theta)
          irho = int(rho_bin * rho / rho_max) # find the rho bin
          hough_space[irho, itheta] += 1 # a vote for the given point


  coord = list()
  # find the most probable lines
  indices = np.argwhere(hough_space > threshold)

  for i in range(indices.shape[0]):

    print(indices[i])

    r, t = np.unravel_index(indices[i], hough_space.shape)

    r_max = r / rho_bin * rho_max 
    t_max = t / theta_bin * theta_max 

    x = np.arange(w)
    y = 1 / math.sin(t_max) * (r_max - x * math.cos(t_max))

    coord.append((r_max,t_max))

    
  hough_space = np.ceil(255 * (hough_space - hough_space.min())/(hough_space.max() - hough_space.min()))

  return hough_space.astype(np.uint8), coord


image_rgb = cv.imread("/Users/catalin/Documents/computer-vision/Labs6/8086.jpg")
image_rgb = cv.GaussianBlur(image_rgb,(5, 5), 0)
img_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')

edges = cv.Canny(img_gray, 100, 200)
plt.imshow(edges, cmap='gray')

hough_response, coordinates = compute_hough_space(edges, 200)
plt.imshow(hough_response, cmap='gray')

plt.imshow(edges, cmap='gray')

for coordinate in coordinates:
  plt.plot(coordinate[0], coordinate[1])