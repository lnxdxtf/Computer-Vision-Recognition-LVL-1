import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions.face_detection import FaceDetection
#basics
v1 = "./Videos/3.mp4"
v0 = 0
cap = cv2.VideoCapture(v0)

pTime = 0
#////////////////////
#mp solutions////////
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection(0.6)



while True:
    success, img = cap.read()
    #// convert bgr to rgb 

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #//points/detection
    results = FaceDetection.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.location_data.relative_bounding_box)]
            #//draw and puting values bounding box // imgH = imgAltura imgW = imgLargur, imgC = imgChanel
            imgH, imgW, imgC = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * imgW), int(bboxC.ymin * imgH), \
                   int(bboxC.width * imgW), int(bboxC.height * imgH)
            cv2.rectangle(img,bbox, (0,0,255), 3)

    #window///
    cTime = time.time() #cTime = current time 
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(600,30),cv2.FONT_HERSHEY_PLAIN, 2,(0,255,59), 2)
    cv2.imshow("face_detect_img",img)
    cv2.waitKey(10) 