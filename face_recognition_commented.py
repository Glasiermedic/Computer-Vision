# This code has extra comments to help explain the process and is updated to reflect changes in OpenCv and python

# make sure you download the xml files and install the latest opencv.

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect(gray, frme): # function that accepts a greyscale and the original frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)
    # this method locates the coordinates of the faces using haar cascade for the face recognition
    # method provides (x,y,w,h) for upper left corner and dimensions
    for (x,y,w,h) in faces: 
        cv2.rectangle(frme, (x,y), (x+w, y+h), (255,0,0), 2) # We pain a rectangle around the face
        rt_gray = gray[y:y+h, x:x+w] # suspect facial region of grayscale image
        rt_color = frame[y:y+h, x:x+w]# applying the suspect facial region to the original image
        eyes = eye_cascade.detectMultiScale(rt_gray, 1.1, 3) # now applying multiscale inside suspect region 
        # to look for the eyes
        for (ex, ey, ew, eh) in eyes: # applied to each eye detected 
            cv2.rectangle(rt_color, (ex, ey), (ex+ew, ey+eh), (0,255, 0), 2) # put rectangle around suspect eyes in suspect face
    return frme

video_capture = cv2.VideoCapture(0) # turning on the video source if set to '0' or the file if location provided

# so far the code is designed for a single image being processed
# the next section will create the process of processing eaach frame of a video stream 

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video to stop hit q', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()



