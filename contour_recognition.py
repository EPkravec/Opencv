import cv2
import matplotlib.pyplot as plt

img = cv2.imread('shar.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

plt.imshow(binary, cmap='gray')
plt.show()

conturs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, conturs, -1, (0, 255, 0), 2)
plt.imshow(img)
plt.show()