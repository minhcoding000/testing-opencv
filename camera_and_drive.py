import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import RPi.GPIO as GPIO
import pigpio


cap = cv2.VideoCapture(0)


GPIO.setmode(GPIO.BOARD)
pi = pigpio.pi()

GPIO.setwarnings(False)

en1 = 4
a1 = 3
a2 = 5

en2 = 14
a3 = 16
a4 = 18

GPIO.setup(a1,GPIO.OUT)               #Actual Pin, not GPIO
GPIO.setup(a2,GPIO.OUT)
GPIO.setup(a3,GPIO.OUT)
GPIO.setup(a4,GPIO.OUT)

pi.set_mode(en2,pigpio.OUTPUT)        #Actual GPIO, not pins
pi.set_mode(en1,pigpio.OUTPUT)

def forward(speed):
    pi.set_PWM_dutycycle(en1,speed)
    pi.set_PWM_dutycycle(en2,speed)
    GPIO.output(a1,GPIO.LOW)
    GPIO.output(a2,GPIO.HIGH)
    GPIO.output(a3,GPIO.HIGH)
    GPIO.output(a4,GPIO.LOW)
    

def stop():
    pi.set_PWM_dutycycle(en2,0)
    pi.set_PWM_dutycycle(en1,0)
    GPIO.output(a1,GPIO.LOW)
    GPIO.output(a2,GPIO.LOW)
    GPIO.output(a3,GPIO.LOW)
    GPIO.output(a4,GPIO.LOW)


def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0] #start height
    y2 = int(y1 *(3/5)) #end height
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1,x2),(y1,y2),1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:
                    left_fit.append((slope,intercept))
                else:
                    right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
        
    return np.array([left_line,right_line])

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150) #weak gradient, and strong gradient
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([(0,height),(1100,height),(1100,height-300),(450,350)])
    mask = np.zeros_like(image)       #black image same size as original image
    cv2.fillPoly(mask,[polygon],255)   #apply white triangle onto mask
    masked_image = cv2.bitwise_and(image,mask) #computing bitwise of two images
    return masked_image


while(True):
    #Capture frame-by-frame
    ret,frame = cap.read()
    lane_image = np.copy(frame)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)  #single degree precision
    if (lines is not None):
        averaged_lines = average_slope_intercept(lane_image,lines)
        line_image = display_lines(lane_image,averaged_lines)
        combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
    
    #Our operations on the frame come here
    #gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    #forward(250);
    #stop();
    
    #Display the resulting frame
    else: 
        cv2.imshow('frame',combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    #When everything done, release the capture
#stop();
#GPIO.cleanup();

cap.release()
cv2.destroyAllWindows()