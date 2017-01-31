import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os.path as path

def calibrate_pts(input_path):
    objpoints = [] # Objects in real world space
    imgpoints = [] # Objects in 2-D space

    nx, ny = 9, 6

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    jpg_files = path.join(input_path, '*.jpg')
    imgs = glob.glob(jpg_files)

    for fname in imgs:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return imgpoints, objpoints


def undistort(img, imgpts, objpts):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img.shape[1::-1], None, None)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
