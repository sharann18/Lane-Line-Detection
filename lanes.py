import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

def canny_conv(c_image):
    gray_img = cv2.cvtColor(c_image, cv2.COLOR_RGB2GRAY) #converts image from RGB to grayscale.
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    canny_img = cv2.Canny(blur_img,50,150)
    return canny_img

def display_image(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    height = image.shape[0] #stores the number of rows in height
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image) #return an array of zeros with the same shape and type as a given array; black color
    cv2.fillPoly(mask, polygons, 255) #used to draw filled polygons like rectangle, triangle, pentagon over an image. 255 ---> white
    #basically we are making a black color mask from the given image and drawing a triangle over it.
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

img = cv2.imread("lane_image.jpg") #loads an image from the specified file, makes an array.
lane_img = np.copy(img)
canny_image = canny_conv(lane_img)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
averaged_lines = average_slope_intercept(cropped_image, lines)
line_image = display_image(lane_img, averaged_lines)
combo_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)

#For a video

# cap = cv2.VideoCapture("lane2.mp4") #loads video
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny_conv(frame)
#     cropped_image = region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#     averaged_lines = average_slope_intercept(cropped_image, lines)
#     line_image = display_image(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow("result", combo_image)
#     if cv2.waitKey(1) == ord('f'): #here, the waitKey is given a value of 1 -> 1 millisecond. Note if 0 is given, we'll have to wait infinitely bw the frames.
#         break
# cap.release() #drops the video file
# cv2.destroyAllWindows()