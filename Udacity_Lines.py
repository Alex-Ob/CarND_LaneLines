import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os
from PIL import Image


def read_image(image_file):
#reading in an image
    image = mpimg.imread(image_file)

    # Need for my imread() settings
    # image = (np.uint8)(image * 255)
    # print(image.dtype)

    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  
    # if you wanted to show a single color channel image called 'gray',
    # for example, call as plt.imshow(gray, cmap='gray')

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def GetMaskByPoly(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    #masked_image = cv2.bitwise_and(img, mask)
    return mask
    
def GetMaskByColor(img, Color):
    """
    Applies an image mask.
    Only keeps the region of the image which is satisfied of color threshold.
    The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    color_thresholds = (img[:,:,0] < Color[0]) \
            | (img[:,:,1] < Color[1]) \
            | (img[:,:,2] < Color[2])
    mask = np.zeros_like(img) # + [255,255,255]
    mask[color_thresholds] = [0,0,0]
    mask[~color_thresholds] = [255,255,255]
    return mask

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * a + img * b + c
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)
    
def TestImages():
    """
    Function checks all files in test_images/ and shows they as image
    """
    import os
    CurrentDir = 'c:/TEXT/ТехЛит/Курсы/Udacity/test_images'
    names = os.listdir(CurrentDir)

    for filename in names:
        #FullFilename = CurrentDir+filename
        FullFilename = os.path.join(CurrentDir, filename)
        print(FullFilename)
        image = mpimg.imread(FullFilename)
        plt.imshow(image)
        plt.pause(1)


# Build a Lane Finding Pipeline
def FindLineOnFrame(image, ShowFlag = False):
    """
    Function detects road lines on the picture and returns these segments
    """
    ysize = image.shape[0]
    xsize = image.shape[1]

    # ############ Color & Region selection
    color_select = np.copy(image)
    region_select = np.copy(image)
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
    gray = grayscale(image)

    #Color masking
    red_threshold = 120
    green_threshold = 120
    blue_threshold = 10
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    ColorMask = GetMaskByColor(image, rgb_threshold)

    Ymid = round(ysize*0.6)
    A = [0, ysize-1]
    B = [round(xsize*0.5), Ymid]
    C = [round(xsize*0.55), Ymid]
    D = [xsize, ysize]

    ROI = np.array( [[A,B,C,D]], dtype=np.int32 )
    
    #ROI mask
    RegionMask = GetMaskByPoly(image, ROI)
    
    #ROI & Color mask
    ColorRegionMask = cv2.bitwise_and(RegionMask, ColorMask)
    
    MaskedImg = cv2.bitwise_and(ColorRegionMask, image)
    
    # Edges finding, given parameters from test Quiz
    kernel_size = 5
    low_threshold = 50
    high_threshold = 150

    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    edges3 = np.dstack((edges, edges, edges))

    # Mask edges
    masked_edges = cv2.bitwise_and(ColorRegionMask, edges3)
    
    # Define the Hough transform parameters
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20

    road_lines = cv2.HoughLinesP(masked_edges[:,:,0], rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    
# Extend lines segments
    
    
#    Xl = road_lines(:,1).index(min(road_lines(:,1)))
#    Xl = road_lines(:,1).index(min(road_lines(:,1)))
    
#    Xmiddle = xsize/2
#    Ymiddle = (Ymid+ysize)/2
    
#    Left_mask = road_lines[:,1] < Xmiddle
#    Bottom_mask = road_lines[:,2] < Xmiddle
#    Xl = min(road_lines(Left_mask,1))
    


    # Draw the lines on the edge image
    draw_lines(line_image, road_lines)
    combo = cv2.addWeighted(masked_edges, 0.8, line_image, 1, 0) 
    
    
    if ShowFlag:
    # Drawing the results
    
    #1st plot
        #Original
        imageROI = np.copy(image)
        cv2.polylines(imageROI,ROI,True,[255,0,0],2)
    
        plt.subplot(2,2,1)
        plt.imshow(imageROI)
        plt.title('Original image')
        plt.tick_params(axis='both', labelsize=0, length = 0)

        plt.subplot(2,2,2)
        plt.imshow(RegionMask)
        plt.title('Region Mask')
        plt.tick_params(axis='both', labelsize=0, length = 0)

        #Color mask
        plt.subplot(2,2,3)
        plt.imshow(ColorMask)
        plt.title('Color Mask')
        plt.tick_params(axis='both', labelsize=0, length = 0)
    
        plt.subplot(2,2,4)
        plt.imshow(MaskedImg)
        plt.title('Color&Region Masked')
        tick_params(axis='both', labelsize=0, length = 0)
        plt.pause(3)

    #2nd plot
        plt.subplot(2,2,1)
        plt.imshow(image)
        plt.title('Original')
        plt.tick_params(axis='both', labelsize=0, length = 0)

    #Add ROI shape
        cv2.polylines(edges3,ROI,True,[255,0,0],10)
    
        plt.subplot(2,2,2)
        plt.imshow(edges3)
        plt.title('Raw edges')
        plt.tick_params(axis='both', labelsize=0, length = 0)

        plt.subplot(2,2,3)
        plt.imshow(masked_edges)
        plt.title('Masked edges')
        plt.tick_params(axis='both', labelsize=0, length = 0)

        plt.subplot(2,2,4)
        plt.imshow(combo)
        plt.title('Lines')
        plt.tick_params(axis='both', labelsize=0, length = 0)
    
    return road_lines
    

def Process_Images():    
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

    CurrentDir = 'test_images'
    names = os.listdir(CurrentDir)
    for filename in names:
        print(filename)

    OutputDir = 'test_images_output'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    for filename in names:
        FullFilename = os.path.join(CurrentDir, filename)
        print(FullFilename)
        image = mpimg.imread(FullFilename)
    
        road_lines = FindLineOnFrame(image)

        line_image = np.zeros_like(image)
        draw_lines(line_image, road_lines)

        # Draw the lines on the edge image
        combo = cv2.addWeighted(image, 0.8, line_image, 1, 0) 

        OutputName = os.path.join(OutputDir, filename)
        plt.imsave(OutputName, combo)

        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title(filename)
        plt.tick_params(axis='both', labelsize=0, length = 0)

        plt.subplot(1,2,2)
        plt.imshow(combo)
        plt.title('The same with Lines')
        plt.tick_params(axis='both', labelsize=0, length = 0)
    
        plt.pause(1)

    
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    Lines = FindLineOnFrame(image)
    
    line_image = np.zeros_like(image)
    draw_lines(line_image, Lines)
    
    # Draw the lines on the edge image
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    
    return combo

def process_video(videofile):
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    import os
    
    
    video_inpath = 'test_videos'
    video_outpath = 'test_videos_output'

    # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    # To do so add .subclip(start_second,end_second) to the end of the line below
    # Where start_second and end_second are integer values representing the start and end of the subclip
    # You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    
    #video_input = video_inpath + "\\" + videofile
    #video_output = video_outpath + "\\" + videofile
    video_input = os.path.join(video_inpath, videofile)
    video_output = os.path.join(video_outpath, videofile)

    if os.path.exists(video_output):
        os.remove(video_output)
        
    video1 = VideoFileClip(video_input)
    video2 = video1.fl_image(process_image) #NOTE: this function expects color images!!
    
    
    #%time video2.write_videofile(video_output, audio=False)
    video2.write_videofile(video_output, audio=False)

    print('HTML:')

    HTML("""
    <video width="960" height="540" controls>
    <source src="{0}">
    </video>
    """.format(video_output))

    print('Done!')
    
#Process_Images()

process_video('solidWhiteRight.mp4')
#process_video('solidYellowLeft.mp4')
