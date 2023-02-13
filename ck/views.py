from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import numpy as np
import mediapipe as mp
import math

def home(request):
    return render(request, 'ck/index.html')

def hand_gesture(request):
    return StreamingHttpResponse(gen_hand_gesture(VideoCamera(0)),
                    content_type='multipart/x-mixed-replace; boundary=frame')

class VideoCamera(object):
    def __init__(self,camera_id):
        self.video = cv2.VideoCapture(camera_id)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        
        frame = self.frame
        frame = cv2.flip(frame,1)

        # Get hand data from the rectangle sub window   
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
        crop_image = frame[100:300, 100:300]
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(crop_image, (3,3), 0)
        
        # Change color-space from BGR -> HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        # Create a binary image with where white will be skin colors and rest is black
        mask = cv2.inRange(hsv, np.array([2,13,26]), np.array([20,255,255]))
        
        # Kernel for morphological transformation    
        kernel = np.ones((5,5))
        
        # Apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask, kernel, iterations = 4) 
        
        # Apply Gaussian Blur and Threshold
        filtered = cv2.GaussianBlur(dilation, (3,3), 0)
        ret,thresh = cv2.threshold(filtered, 127, 255, 0)
        
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        
        try:

            # Find contour with maximum area
            contour = max(contours, key = lambda x: cv2.contourArea(x))

            epsilon = 0.0005*cv2.arcLength(contour,True)
            approx= cv2.approxPolyDP(contour,epsilon,True)
            
            # Find convex hull
            hull = cv2.convexHull(contour)

            # Define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(contour)
            
            # find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100

            # find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx,hull)

            # count_defects = no. of defects
            count_defects = 0

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
            # tips) for all defects

            # Finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                # distance between point and convex hull
                d=(2*ar)/a

                # apply cosine rule her
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # if angle > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d > 30:
                    count_defects += 1
                    cv2.circle(crop_image,far,3,[255,0,0],-1)
                # draw lines around hand
                cv2.line(crop_image,start,end,[0,255,0],2)

            # print corresponding gestures which are in their ranges
            count_defects += 1
            if count_defects==1:
                if arearatio<8:
                    if arearatio > 6.45:
                        cv2.putText(frame,"STOP", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    else:
                        cv2.putText(frame,"ZERO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                elif arearatio > 15.75 and arearatio < 19.5:
                    cv2.putText(frame,"GOOD", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    
                else:
                    cv2.putText(frame,"ONE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        
            elif count_defects==2:
                cv2.putText(frame,"TWO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                
            elif count_defects==3:
                if arearatio > 19.5:
                    cv2.putText(frame,"OK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                else:
                    cv2.putText(frame,"THREE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        
            elif count_defects==4:
                cv2.putText(frame,"FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                
            elif count_defects==5:
                cv2.putText(frame,"FIVE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                
            else:
                pass
        except:
            pass

        
        # Show required image
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen_hand_gesture(camera):
    while True:    
        frame = camera.get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: img/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
      