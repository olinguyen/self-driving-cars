import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from calibrate_camera import *
from thresholding import *
from perspective_transform import *


calibrate_folder = "./camera_cal"
imgpoints, objpoints = calibrate_pts(calibrate_folder)

# Class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque()
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # num of iterations to keep
        self.to_keep = 10

    def smooth_fit(self):
        self.recent_xfitted.append(self.allx)

        if len(self.recent_xfitted) > self.to_keep:
            self.recent_xfitted.popleft()

        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(self.bestx, self.ally, 2)


    def compute_radius_curvature(self):
        y_eval = np.max(self.ally) / 2

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally * ym_per_pix, self.bestx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        return curverad


    def compute_position(self):	
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        y = np.max(self.ally)
        position = self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]

        pixels_off_center = int(position - (img_size[0]/2))
        self.line_base_pos = xm_per_pix * pixels_off_center
        return self.line_base_pos


class LaneDetector():
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.center = 0
        self.M = None
        self.Minv = None
    
    def pipeline(self, img):
        undistorted = undistort(img, imgpoints, objpoints)

        binary_output = combine_thresh(undistorted)

        if self.M is None:
            self.M, self.Minv = compute_perspective_matrix(img)

        binary_warped = apply_warp(binary_output, self.M)

        if self.left_line.detected == True:
            self.left_line, self.right_line = detect_lines(binary_warped,
                    self.left_line, self.right_line)
        else:
            self.left_line, self.right_line = detect_lines_blind(binary_warped)
            self.left_line.detected = True
            self.right_line.detected = True

        self.left_line.smooth_fit();
        self.right_line.smooth_fit();

        left_curverad = self.left_line.compute_radius_curvature()
        right_curverad = self.right_line.compute_radius_curvature()
        curverad = (left_curverad + right_curverad) / 2

        position = self.left_line.compute_position() + self.right_line.compute_position() 

        result = draw_lanes(img, binary_warped, self.Minv, self.left_line, self.right_line)


        result = draw_info(result, curverad, position)

        return result


def detect_lines(img, left_line, right_line, smooth=False):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    if smooth == True and left_line is not None:
        left_fit = left_line.best_fit
        right_fit = right_line.best_fit
    else:
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) 
        + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) 
        + left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) 
        + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) 
        + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    """
    print(left_fit, leftx.shape, leftx.shape)
    print(right_fit, rightx.shape, righty.shape)
    print(right_line.current_fit)
    """

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Update Line objects & extrapolate on all the line
    left_line.current_fit = left_fit 
    right_line.current_fit = right_fit

    left_line.ally = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_line.allx = left_fit[0]*left_line.ally**2 + left_fit[1]*left_line.ally + left_fit[2]

    right_line.ally = np.linspace(0, img.shape[0]-1, img.shape[0])
    right_line.allx = right_fit[0]*right_line.ally**2 + right_fit[1]*right_line.ally + right_fit[2]

    if not leftx.any() or not rightx.any():
        left_line.detected = False
        right_line.detected = False


    return left_line, right_line


def detect_lines_blind(binary_warped):
    left_line = Line()
    right_line = Line()

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    #plt.plot(histogram)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Width of the windows +/- margin
    margin = 100
    
    # Minimum number of pixels found to recenter window
    minpix = 50

    # Empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # update Line objects & extrapolate on all the line
    left_line.current_fit = left_fit 
    right_line.current_fit = right_fit

    left_line.ally = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_line.allx = left_fit[0]*left_line.ally**2 + left_fit[1]*left_line.ally + left_fit[2]

    right_line.ally = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    right_line.allx = right_fit[0]*right_line.ally**2 + right_fit[1]*right_line.ally + right_fit[2]

    if leftx.any() or rightx.any():
        left_line.detected = True
        right_line.detected = True

    return left_line, right_line


def draw_lanes(img, binary_warped, Minv, left_line, right_line, smooth=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    fity = left_line.ally

    if smooth == True and left_line.bestx is not None:
        fit_leftx = left_line.bestx
        fit_rightx = right_line.bestx
    else:
        fit_leftx = left_line.allx
        fit_rightx = right_line.allx

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result


def draw_info(img, curvature, position):
    text_color = (255,255,255)
    text_str = 'Radius of curvature: %dm' % (curvature)
    cv2.putText(img, text_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)

    if (position) < 0:
        text_str = 'Vehicle is %5.2fm left of center'%(-position)
    elif (position) > 0:
        text_str = 'Vehicle is %5.2fm right of center'%(position)
    else:
        text_str = 'Vehicle is at center'
    cv2.putText(img, text_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color)
    return img
