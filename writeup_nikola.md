## Writeup for Advanced Lane Lines Project: Nikola Dordevic


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/output_calibration1.jpg "Undistorted"
[image2_org]: ./test_images/test2.jpg "Road Transformed"
[image2]: ./output_images/undistorted_test2.jpg "Road Transformed"
[image3]: ./output_images/binarized.jpg "Binary Example"
[image4]: ./output_images/warped.jpg "Warp Example"
[image6]: ./output_images/final_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first part of cells of the IPython notebook located in `CameraCalibration.ipynb`. As a sanity check, I followed [tutorial](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html) on calibration as well.  

First, all images are loaded using `glob` package since the file naming was done i na way that allows for batch loading.

For every loaded image, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function:
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

Calibration parameters are saved for further usage in the project:
```python
# Store camera calibration

res_dict = dict(ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

fname = 'calibrationSettings.pk'
outfile = open(fname,'wb')
pickle.dump(res_dict,outfile)
outfile.close()
```

  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)
In order to have more controlled environment for the pipeline and the batch processing of frames, entire functionality after camera calibration is implemented into the module `lane_finder.py`. I created the class `LaneFinder` into which image transformations are implemented as methods (some of them as `@staticmethod`).

`LaneFinder` class is initialized using the following code:

```python

from lane_finder.LaneFinder import LaneFinder

ll = LaneFinder(
    calibration_file='./calibrationSettings.pk', 
    binarization_settings_file='./binarization_settings.json', 
    perspective_settings_file='perspective_settings.json', 
    road_settings_file='./road_settings.json')

```
with `calibration_file` having parameters of camera calibraiton, `binarization_settings_file` is `json` file having parameters for binarization (color, gradient and direction of gradient parameters), `perspective_settings_file` is `json` file that stores settings for changing perspective of the image (warp/unwarp) and `road_settings_file` describes relation between pixel-distance and real-world distance.


Processing of images (or frames) fed to the `LaneFinder` are combined in a single method (pipeline) called `LaneFinder.process_frame`. First couple of lines of this method defined in `lane_finder/LaneFinder.py`:

```python
def process_frame(self, img) -> None:
    # Check if color order is OK with pyVideo
    # If not, change to RGB, since all methods are made for RGB
    self.output_image = None    # Just to make sure
    self.set_image(img) # Stores original image and applies distortion correction
    self.binarize()     # Uses settings defined at init to binarize the input image
    self.warp_binarized()   # Change perspective for lane finding
...
```


#### 1. Provide an example of a distortion-corrected image.

Distortion-correction was performed on all test images using following code in Jupyter notebook:
``` python
from lane_finder.LaneFinder import LaneFinder

ll = LaneFinder(
    calibration_file='./calibrationSettings.pk', 
    binarization_settings_file='./binarization_settings.json', 
    perspective_settings_file='perspective_settings.json', 
    road_settings_file='./road_settings.json')

tst_fnames = os.listdir('./test_images/')

for ii,ff in enumerate(tst_fnames):
    img = plt.imread(f'./test_images/{ff}')
    ll.process_frame(img);
    fig, ax = plt.subplots(nrows=2,figsize=(12,9))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(ll.corrected_image)
    ax[1].set_title('Undistorted Image')
    out_name_base = ff.split('.')[0]
    fig.tight_layout()
    ll.reset_frame()
    fig.savefig(f'./output_images/undistorted_{out_name_base}.jpg')
```

The actual distprtion correction happens within `ll.process_frame(img)` using a method `correct_distortion`:

```python

def correct_distortion(self) -> None:
    mtx = self.mtx
    dist = self.dist
    self.corrected_image = cv2.undistort(self.original_image, mtx, dist, None, mtx)

```


To demonstrate this step, one of the distortion-corrected images (`test2.jpg`) is shown:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. As shown above, this is done in the following method:

```python
def binarize(self) -> None:
    ksize= self.binarization_settings['kernel_size']
    bs = self.binarization_settings.copy()
    gradx = LaneFinder.abs_sobel_thresh(self.corrected_image, color_channel=bs['gradient_x']['channel'], orient='x', sobel_kernel=ksize, thresh=tuple(bs['gradient_x']['threshold']))
    dir_binary = LaneFinder.dir_threshold(self.corrected_image,color_channel=bs['gradient_direction']['channel'], sobel_kernel=ksize, thresh=bs['gradient_direction']['threshold_rad'])
    sbinary = LaneFinder.threshold_color(self.corrected_image,color_channel=bs['color_threshold']['color'][0], threshold=bs['color_threshold']['threshold'][0])
    lbinary = LaneFinder.threshold_color(self.corrected_image,color_channel=bs['color_threshold']['color'][1], threshold=bs['color_threshold']['threshold'][1])
    res = (dir_binary & gradx) | sbinary | lbinary
    self.binarized_image = res.copy()
```
and content of `./binarization_settings.json` is:
```json
{
    "kernel_size": 3,
    "gradient_x":{
        "channel": "S",
        "threshold": [8,200]
    },
    "gradient_direction":{
        "channel": "S",
        "threshold_degree": [30,60]
    },
    "color_threshold": {
        "color": ["S","L"],
        "threshold":[[150,255], [200,255]]
    }
}
```

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform was done using `LaneFinder` methods:

```python

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

@staticmethod
def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
```

Where `offset` and `source` points are parameters defiend at the `__init__()` method.



![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Rest of the `process_frame` method (line #230-249) contains lane finding algorithm. 
```python

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

# To make sure that this is noted in the history as well
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

```
As seen above, there are two main methods:
-   ```self.find_lanes_window()```
    - This method is define at line #89 (`LaneFinder.py`) and uses the sliding windows approach shown in course videos.
-   `self.find_lanes_from_previous()`
    - This method uses previous lane found as a base search and is defined at line #180.

After the lanes are fit, the separation between lanes is measured along x-axis. If the starndard deviation for these measurements, the lanes are discarted and mean of previous `PREV_FRAME_MIN` images is taken as a new set of poitns which is than fitted again.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated on line #293.

```python

R_curve_left, R_curve_right = self.measure_curvature_pixels(yy=yy_scaled, left_fitx=lx, right_fitx=rx)

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


```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/outputProjectVideo.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my approach, one of the things that could easily be better is binarization that is more robust.

On top of that, the 'robustness' test that I have at the moment is connected to how much the lanes are parallel. Although this covers a lot of possible catastrophic errors, it is not robust enough in my oppinion.

Also, as a back-up when track is not found, mean value of 5 previous lanes are used. This might be problematic in sharp turns and large speeds.
