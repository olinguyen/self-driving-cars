import numpy as np
import cv2

img_size = 1280, 720

"""
src = np.float32(
    [[(img_size[0] / 2) - 30, img_size[1] / 2 + 100],
    [((img_size[0] / 6) + 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 90, img_size[1]],
    [(img_size[0] / 2 + 90), img_size[1] / 2 + 100]])

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
"""
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])


def warp(img):
    img_size = img.shape[1], img.shape[0] # if color image, then use img.shape[1::-1]

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped, src, dst

def compute_perspective_matrix(img):
    img_size = img.shape[1], img.shape[0] # if color image, then use img.shape[1::-1]

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def apply_warp(img, M):
    img_size = img.shape[1], img.shape[0] # if color image, then use img.shape[1::-1]
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped

