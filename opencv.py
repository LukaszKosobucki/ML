import cv2 as cv
import numpy as np
import cv2
import glob

# # termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6 * 9, 3), np.float32)
# objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# # Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
# img = cv2.imread('calib_result.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Find the chess board corners
# ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
# # If found, add object points, image points (after refining them)
# if ret == True:
#     objpoints.append(objp)
#     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#     imgpoints.append(corners)
#     # Draw and display the corners
#     cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

import matplotlib.pyplot as plt

# Test undistortion on an image
img = cv2.imread('calib_result.jpg')
img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('undistorted.jpg', dst)

# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('calib_result.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('calibresult.png', dst)

# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)