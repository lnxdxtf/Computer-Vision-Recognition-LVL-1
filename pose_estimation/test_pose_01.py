import cv2 
import mediapipe as mp
import time
import numpy as np

fps_1 = 0 #previoustime
fps_2 = 0 #currentytime

v1 = 'C:/Users/gabri/Desktop/DEV/Projetos_GitHub/Computer_Vision_Recognition_LVL_1/pose_estimation/PoseVideos/3.mp4' 
#mude o ultimo arquivo (1.mp4 | 2.mp4 | 3.mp4)
#!!! ARRUMAR NO GIT !!!

cap = cv2.VideoCapture(v1)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

                                                        
while True:
    success, img = cap.read()

    #converter img para o processo, pq mediapipe usa rgb e n bgr
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    #processo 
    results = pose.process(imgRGB) 
    """results.pose_landmarks     - para pegar o resultado x,y,z e a visibilidade.
    EX: print(results.pose_landmarks)
    """
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, points in enumerate(results.pose_landmarks.landmark): #lm = points
            alt,larg,c = img.shape
            print("ID:",id,"\n Points:\n",points)
            cx, cy = int(points.x * larg), int(points.y * alt)
            cv2.circle(img,(cx,cy),5,(0,255,80),cv2.FILLED)

    #TELA/JANELA
    fps_2 = time.time()
    fps = 1/(fps_2-fps_1)
    fps_1 = fps_2
    #theme 
    cv2.putText(img,'TEST_POSE',(300,20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255), 2)
    cv2.putText(img,'FPS:',(580,20),cv2.FONT_HERSHEY_PLAIN,1,(209,0,178), 2)
    cv2.putText(img,str(int(fps)),(615,20),cv2.FONT_HERSHEY_PLAIN, 1,(209,0,178), 2)
    cv2.imshow("video", img)


    cv2.waitKey(1)

