import pickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

class LaneFinder:
    '''
    class LaneFinder for locating lanes and tracks on images from the camera located on a car.
    '''
    PREV_FRAME_MIN = 5

    def __init__(self, calibration_file:str, binarization_settings_file:str, perspective_settings_file:str, road_settings_file:str) -> None:
        """
        Init is done only with calibration files and default values for pipeline processing.
        """
        # Load calibration of the camera used
        calib_dict = self._load_camera_calibration(calibration_file=calibration_file)
        self.mtx = calib_dict['mtx']
        self.dist = calib_dict['dist']
        # For image processing history of each image/frame
        self.original_image = None
        self.corrected_image = None
        self.binarized_image = None
        self.warped_image = None
        self.output_image = None

        # To change perspective
        self.src = None
        self.dst = None

        # class members for batch processing
        self._prev_fit = None
        # For production code, I would make here a FitHistory class with enhanced features
        self._fit_history = []
        self._rcurve_history = []
        # If we need to revert to windows search
        # Not implemented
        self._reset_windows = False

        # Load settings for easier customization of different LaneFinders
        self.binarization_settings = self._load_binarization_settings(binarization_settings_file)
        self.perspective_settings = self._load_perspective_settings(perspective_settings_file)
        self.x_m_pix, self.y_m_pix = self._load_road_settings(road_settings_file)

    def load_image(self, image_path) -> None:
        self.original_image = plt.imread(fname=image_path)
        self.correct_distortion()
        return self

    def set_image(self, image) -> None:
        self.original_image = image
        self.correct_distortion()
        return self

    def correct_distortion(self) -> None:
        mtx = self.mtx
        dist = self.dist
        self.corrected_image = cv2.undistort(self.original_image, mtx, dist, None, mtx)

    def binarize(self) -> None:
        ksize= self.binarization_settings['kernel_size']
        bs = self.binarization_settings.copy()
        gradx = LaneFinder.abs_sobel_thresh(self.corrected_image, color_channel=bs['gradient_x']['channel'], orient='x', sobel_kernel=ksize, thresh=tuple(bs['gradient_x']['threshold']))
        dir_binary = LaneFinder.dir_threshold(self.corrected_image,color_channel=bs['gradient_direction']['channel'], sobel_kernel=ksize, thresh=bs['gradient_direction']['threshold_rad'])
        sbinary = LaneFinder.threshold_color(self.corrected_image,color_channel=bs['color_threshold']['color'][0], threshold=bs['color_threshold']['threshold'][0])
        lbinary = LaneFinder.threshold_color(self.corrected_image,color_channel=bs['color_threshold']['color'][1], threshold=bs['color_threshold']['threshold'][1])
        res = (dir_binary & gradx) | sbinary | lbinary
        self.binarized_image = res.copy()

    def warp_binarized(self, *args) -> None:
        assert self.binarized_image is not None
        img_size = [self.corrected_image.shape[1], self.corrected_image.shape[0]]

        src = np.float32(self.perspective_settings['src_points'])
        offset = self.perspective_settings['offset_rel'] * img_size[0]

        dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                            [img_size[0]-offset, img_size[1]], 
                            [offset, img_size[1]]])

        self.src = src
        self.dst = dst

        self.warped_image = LaneFinder.warper(self.binarized_image, src=src, dst=dst)


    def find_lanes_window(self, nwindows=9, margin=100, minpix=50) -> None:
        warped_img = self.warped_image

        out_img = np.dstack((warped_img,warped_img,warped_img))*255
        histogram = np.sum(warped_img[warped_img.shape[0]//2:,:],axis=0)

        # Starting point for the middle of the track
        midpoint = np.int32(histogram.shape[0]//2)

        # Base points for the lane boundaries
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Define the windows and hyperparameters


        window_height = warped_img.shape[0]//nwindows
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base


        tmp = out_img.copy()
        
        # Create empty lists to receive left and right lane pixel indices
        leftx = []
        lefty = []
        rightx = []
        righty = []

        for iw in range(nwindows):

            # select active pixels within a window
            leftx_old = leftx_current
            rightx_old = rightx_current

            wy_low,wy_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high = \
                LaneFinder.get_window_range(tmp, window_height, margin, leftx_old, rightx_old, iw)


            # find active pixels indices
            fleft = (nonzeroy >= wy_low) & (nonzeroy < wy_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)
            fright = (nonzeroy >= wy_low) & (nonzeroy < wy_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)

            tmp_leftx = nonzerox[fleft]
            tmp_lefty = nonzeroy[fleft]
            tmp_rightx = nonzerox[fright]
            tmp_righty = nonzeroy[fright]


            leftx.append(tmp_leftx)
            lefty.append(tmp_lefty)

            rightx.append(tmp_rightx)
            righty.append(tmp_righty)



            if len(tmp_leftx) > minpix:
                leftx_current = np.int32(np.mean(tmp_leftx))

            if len(tmp_rightx) > minpix:
                rightx_current = np.int32(np.mean(tmp_rightx))

            # Draw the windows on the visualization image
            cv2.rectangle(tmp,(win_xleft_low,wy_low),
            (win_xleft_high,wy_high),(0,255,0), 2) 
            cv2.rectangle(tmp,(win_xright_low,wy_low),
            (win_xright_high,wy_high),(0,255,0), 2)

        leftx = np.concatenate(leftx) 
        lefty = np.concatenate(lefty) 
        rightx = np.concatenate(rightx) 
        righty = np.concatenate(righty)

        lane_pixels = {
            'leftx': leftx,
            'lefty': lefty,
            'rightx': rightx,
            'righty': righty
        }

        left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs = LaneFinder.fit_lanes(self.warped_image, leftx, rightx, lefty, righty)
        
        return tmp, lane_pixels, left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs

    def find_lanes_from_previous(self, margin=100) -> None:
        
        warped_img = self.warped_image.copy()
        out_img = np.dstack((warped_img,warped_img,warped_img))*255
        tmp = out_img.copy()

        active_pixels = self.warped_image.nonzero()
        
        nonzerox = np.array(active_pixels[1])
        nonzeroy = np.array(active_pixels[0])

        leftx_prev = np.polyval(self._prev_fit['left_coefs'], nonzeroy)
        rightx_prev = np.polyval(self._prev_fit['right_coefs'], nonzeroy)

        filter_id_left = (nonzerox > (leftx_prev-margin)) & (nonzerox < (leftx_prev+margin))
        filter_id_right = (nonzerox > (rightx_prev-margin)) & (nonzerox < (rightx_prev+margin))

        # Find lane pixels
        leftx = nonzerox[filter_id_left]
        lefty = nonzeroy[filter_id_left] 
        rightx = nonzerox[filter_id_right]
        righty = nonzeroy[filter_id_right]
        
        lane_pixels = {
            'leftx': leftx,
            'lefty': lefty,
            'rightx': rightx,
            'righty': righty
        }

        # Fit polynomial
        left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs = LaneFinder.fit_lanes(self.warped_image, leftx, rightx, lefty, righty)

        return tmp, lane_pixels, left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs


    def process_frame(self, img) -> None:
        # Check if color order is OK with pyVideo
        # If not, change to RGB, since all methods are made for RGB
        self.output_image = None    # Just to make sure
        self.set_image(img) # Stores original image and applies distortion correction
        self.binarize()     # Uses settings defined at init to binarize the input image
        self.warp_binarized()   # Change perspective for lane finding

        # update this to have a selection where previoys runs can be taken into account
        if self._prev_fit is None:
            tmp, lane_pixels, left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs = self.find_lanes_window()
        else:
            tmp, lane_pixels, left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs = self.find_lanes_from_previous()

        # CHECK LANE
        # If there is too much deviations between lane points, abort this lane and take previous
        lane_deviation_x = np.std(right_fitx - left_fitx)
        if  lane_deviation_x > 25:
            # abort this lane and take from history
            if self.PREV_FRAME_MIN > len(self._fit_history):
                prev_coefs = self._fit_history.copy()
            else:
                prev_coefs = self._fit_history[-self.PREV_FRAME_MIN:]

            # ARTIFICIALLY ADD POINTS BY TAKING MEAN VALUE
            xl_points=np.array(list(map(lambda x: np.polyval(x['left_coefs'],yy), prev_coefs)))
            xr_points=np.array(list(map(lambda x: np.polyval(x['right_coefs'],yy), prev_coefs)))

            left_fit_coefs = np.polyfit(yy,xl_points.mean(axis=0),2)
            right_fit_coefs = np.polyfit(yy,xr_points.mean(axis=0),2)

            # To make sure that this is noten in the history as well
            left_fitx = np.polyval(left_fit_coefs, yy)
            right_fitx = np.polyval(right_fit_coefs, yy)


        lane = np.zeros_like(self.original_image)

        pts = np.vstack((np.flipud(np.transpose([right_fitx,yy])),np.transpose([left_fitx,yy])))
        cv2.fillPoly(lane, np.int_([pts]), (200, 0,200))


        leftLinePts = np.transpose([left_fitx,yy])
        leftLinePts = leftLinePts.reshape((-1,1,2))

        rightLinePts = np.transpose([right_fitx,yy])
        rightLinePts = rightLinePts.reshape((-1,1,2))

        cv2.polylines(lane,np.int_([leftLinePts]),False,(0,255,255),thickness=12)
        cv2.polylines(lane,np.int_([rightLinePts]),False,(0,255,255),thickness=12)
        # cv2.polylines(lane,np.int_([pts]),True,(255,255,255),thickness = 3)


        M = cv2.getPerspectiveTransform(self.src, self.dst)
        M_inv = np.linalg.inv(M)



        img_size = (lane.shape[1], lane.shape[0])
        lane_unwarped = cv2.warpPerspective(lane, M_inv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        wtd = cv2.addWeighted(self.original_image.copy(), 1, lane_unwarped, 0.5, 0.2)

        self.output_image = wtd
        self.original_image = None

        # Scale values for radius of curvature
        rescale = lambda mx,my,p,y: mx / (my ** 2) *p[0]*(y**2)+(mx/my)*p[1]*y+p[2]

        lcoef_rescaled = np.polyfit(yy*self.y_m_pix, left_fitx*self.x_m_pix,2)
        rcoef_rescaled = np.polyfit(yy*self.y_m_pix, right_fitx*self.x_m_pix,2)

        yy_scaled = yy*self.y_m_pix

        lx = np.polyval(lcoef_rescaled, yy_scaled)
        rx = np.polyval(rcoef_rescaled, yy_scaled)
        
        R_curve_left, R_curve_right = self.measure_curvature_pixels(yy=yy_scaled, left_fitx=lx, right_fitx=rx)

        # add to history
        self._prev_fit = {
            'left_coefs': left_fit_coefs,
            'right_coefs': right_fit_coefs
        }

        self._fit_history.append(self._prev_fit)
        self._rcurve_history.append([R_curve_left, R_curve_right])

        #write some text

        # Radius of curvature
        txt = f"R_left: {R_curve_left}, R_right: {R_curve_right}"
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,50)
        fontScale              = 1.2
        fontColor              = (255,255,255)
        lineType               = 2
        cv2.putText(wtd,txt, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # Car location

        # pixel location of middle of the lane
        mid_lane_px = 0.5*(left_fitx[-1] + right_fitx[-1])
        mid_img = self.warped_image.shape[1] // 2

        dist = (mid_img - mid_lane_px)*self.x_m_pix
        if dist > 0:
            whr='right'
        else:
            whr='left'

        txt_car = f'Car is {abs(dist)} m to the {whr}'
        cv2.putText(wtd,txt_car, 
        (10,100), 
        font, 
        fontScale,
        fontColor,
        lineType)

        return wtd.copy()

    def reset_frame(self) -> None:
        self._prev_fit = None



    # Private methods
    def _load_camera_calibration(self, calibration_file) -> dict:
        infile = open(calibration_file,'rb')
        calib_dict = pickle.load(infile)
        infile.close()
        return calib_dict

    def _load_binarization_settings(self, binarization_settings) -> dict:
        infile = open(binarization_settings)
        bs = json.load(infile)
        infile.close()
        # There should be assertion here!
        thr_angle = np.deg2rad(np.array(bs['gradient_direction']['threshold_degree']))
        bs['gradient_direction']['threshold_rad'] = tuple(thr_angle)
        tt_color = list(map(lambda x: tuple(x), bs['color_threshold']['threshold']))
        bs['color_threshold']['threshold'] = tt_color.copy()
        return bs

    def _load_perspective_settings(self, perspective_settings) -> dict:
        infile = open(perspective_settings)
        pps = json.load(infile)
        infile.close()
        return pps

    def _load_road_settings(self, road_settings_file) -> None:
        infile = open(road_settings_file)
        rs = json.load(infile)
        return rs['x_m_pix'], rs['y_m_pix']


    @staticmethod
    def threshold_color(img, color_channel, threshold = (0, 255)):
        """
        Performs threshold on an image (assumes that img is loaded with plt.imread).
        Color channel that is accepted must be one of RGB or HLS channels.
        
        Output is binarized image with ones where the threshold conditions are satisfied.
        """
        rgb_channels = ['R', 'G', 'B']
        hls_channels = ['H', 'L', 'S']
        
        assert((color_channel in rgb_channels) | (color_channel in hls_channels))
        
        if color_channel in rgb_channels:
            ch_id = rgb_channels.index(color_channel)
            img_color_channel = img[:,:,ch_id]
        else:
            img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            ch_id = hls_channels.index(color_channel)
            img_color_channel = img_hls[:,:,ch_id]
        
        binary_color = np.zeros_like(img_color_channel)
        binary_color[(img_color_channel > threshold[0]) & (img_color_channel <= threshold[1])] = 1
        return binary_color

    @staticmethod
    def channel_picker(img, color_channel):
        """
        Helper function for picking a single channel out of image
        """
        rgb_channels = ['R', 'G', 'B']
        hls_channels = ['H', 'L', 'S']
        
        assert((color_channel in rgb_channels) | (color_channel in hls_channels))
        
        if color_channel in rgb_channels:
            ch_id = rgb_channels.index(color_channel)
            img_color_channel = img[:,:,ch_id]
        else:
            img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            ch_id = hls_channels.index(color_channel)
            img_color_channel = img_hls[:,:,ch_id]
        return img_color_channel

    @staticmethod
    def abs_sobel_thresh(img, color_channel, orient='x', sobel_kernel=3, thresh=(0,255)):
        
        # Apply the following steps to img
        # 1) take a channel
        gray = LaneFinder.channel_picker(img=img, color_channel=color_channel)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient=='x':
            sobel_tmp = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        if orient=='y':
            sobel_tmp = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel_tmp)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        abs_sobel_scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(abs_sobel_scaled)
        binary_output[(abs_sobel_scaled >= thresh[0]) & (abs_sobel_scaled <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def mag_thresh(img, color_channel, sobel_kernel=3, mag_thresh=(0, 255)):
        
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = LaneFinder.channel_picker(img=img, color_channel=color_channel)
        # 2) Take the gradient in x and y separately
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # 3) Calculate the magnitude
        sobel_mag = np.hypot(sobelx,sobely)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        sobel_mag_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))
        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(sobel_mag_scaled)
        binary_output[(sobel_mag_scaled >= mag_thresh[0]) & (sobel_mag_scaled <= mag_thresh[1])] =1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def dir_threshold(img, color_channel, sobel_kernel=3, thresh=(0, np.pi/2)):
        
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = LaneFinder.channel_picker(img=img, color_channel=color_channel)
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_slope = (np.arctan2(sobely, sobelx))
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(gray)
        binary_output[(grad_slope >= thresh[0]) & (grad_slope <= thresh[1])] =1
        # 6) Return this mask as your binary_output image
        return binary_output


    @staticmethod
    def warper(img, src, dst):

        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped


    @staticmethod
    def get_window_range(img, window_height, margin, leftx_current, rightx_current, iw):
        wy_low = img.shape[0] - (iw+1)*window_height
        wy_high = img.shape[0] - iw*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        return [wy_low,wy_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high]

    @staticmethod
    def fit_lanes(img, leftx, rightx, lefty, righty):
        yy = np.linspace(0,img.shape[0]-1,img.shape[0])
        
        left_fit_coefs = np.polyfit(lefty,leftx,2)
        right_fit_coefs = np.polyfit(righty, rightx,2)
        
        left_fitx = np.polyval(left_fit_coefs,yy)
        right_fitx = np.polyval(right_fit_coefs,yy)
        
        return left_fitx, right_fitx, yy, left_fit_coefs, right_fit_coefs

    @staticmethod
    def measure_curvature_pixels(yy, left_fitx, right_fitx):
        '''
        Calculates the curvature of polynomial functions in pixels.
        Based on functions in lession quizes.
        '''
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(yy)
        
        get_curverad = lambda x: ((1+(2*x[0]*y_eval + x[1])**2)**1.5 )/ (np.abs(2*x[0]))
        left_curverad = get_curverad(left_fitx)
        right_curverad = get_curverad(right_fitx)
        
        return left_curverad, right_curverad

