import cv2 
import mediapipe as mp
import time
import numpy as np
from tqdm import tqdm
from my_pose_modules import pose_module as pm



fps_1 = 0 #previoustime
fps_2 = 0 #currentytime
#'C:/Users/gabri/Desktop/DEV/Projetos_GitHub/Computer_Vision_Recognition_LVL_1/pose_estimation/PoseVideos/1.mp4'
#v1 =  '/pose_estimation/PoseVideos/4.mp4' #1.mp4 | 2.mp4 | 3.mp4 |4.mp4 | 5.mp4
v1 =  './Videos/1.mp4' #1.mp4 | 2.mp4 | 3.mp4 |4.mp4 | 5.mp4
v0 = 0 # troque v1 por v0 para webcam 
cap = cv2.VideoCapture(v1)

detPose = pm.poseDetector()

pbar = tqdm()
a = -0

while True:

    success, img = cap.read()
    if success:
        
        detPose.findPose(img)
        detPose.getPosePosition(img, getPoints=True) #getpoints permite numerar os 33 pontos do corpo de acordo com o arquivo landmarks_pose.png


        fps_2 = time.time()
        fps = 1/(fps_2-fps_1)
        fps_1 = fps_2

        cv2.putText(img,'TEST_POSE',(300,20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255), 2)
        cv2.putText(img,'FPS:',(580,20),cv2.FONT_HERSHEY_PLAIN,1,(209,0,178), 2)
        cv2.putText(img,str(int(fps)),(615,20),cv2.FONT_HERSHEY_PLAIN, 1,(209,0,178), 2)
        cv2.imshow("video", img)
        cv2.waitKey(1)

        pbar.set_description(f"Running")
        pbar.update(a)
        a = a+1
        
    else:
        print()
        pbar.close()
        break
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
cap.release()
cv2.destroyAllWindows()