import numpy as np
import cv2

def warp(img):
    img_size = img.shape[1], img.shape[0] # if color image, then use img.shape[1::-1]

    src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 80), img_size[1]],
        [(img_size[0] * 5 / 6) + 105, img_size[1]],
        [(img_size[0] / 2 + 105), img_size[1] / 2 + 100]])

    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped, src, dst
