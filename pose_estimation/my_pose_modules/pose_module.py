from pathlib import Path
import cv2 
import mediapipe as mp
import time
import numpy as np
from tqdm import tqdm

class poseDetector():
    def __init__(self, mode= False, model= 1, smoothLm= True,
            enableSeg= False, smoothSeg= True, detCon= 0.5, trackCon= 0.5):
        
        self.mode = mode
        self.model = model
        self.smoothLm = smoothLm
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.model,self.smoothLm,
                                    self.enableSeg,self.smoothSeg,self.detCon,
                                    self.trackCon)
                                           

    def findPose(self, img, draw= True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS,
                                        self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2,
                                        circle_radius=1),#pontos
                                        self.mpDraw.DrawingSpec(color=(255,255,0), thickness=2,
                                        circle_radius=0)#linhas
                                        )
        return img
    
    def getPosePosition(self, img, draw= True, getPoints=True, getPrint=False):
        pointsList = []
        if self.results.pose_landmarks:
            for id, points in enumerate(self.results.pose_landmarks.landmark): #lm = points
                alt,larg,c = img.shape
                #print("ID:",id,"\n Points:\n",points)
                cx, cy = int(points.x * larg), int(points.y * alt)
                pointsList.append([id,cx,cy])
                if getPoints:
                    cv2.putText(img,str(int(id)),(cx+10,cy+10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
                    if getPrint:
                        print(f"=========================={id}============================\n"
                        ,pointsList,"\n=======================================================")
                # posso usar cx e cy como coordenadas para desenhar 
                #cv2.circle(img,(cx,cy),5,(0,255,80),cv2.FILLED)

"""    código para testar o módulo --- !TESTANDO TDQM - PBAR !

def main():
    fps_1 = 0 #previoustime
    fps_2 = 0 #currentytime

    v1 = 'C:/Users/gabri/Desktop/DEV/Projetos_GitHub/Computer_Vision_Recognition_LVL_1/pose_estimation/PoseVideos/1.mp4' 
    cap = cv2.VideoCapture(v1)

    detPose = poseDetector()

    
    pbar = tqdm()
    a = -0
    while True:

        success, img = cap.read()
        if success:

            
            detPose.findPose(img)
            detPose.getPosePosition(img, getPoints=True)

            
            
            fps_2 = time.time()
            fps = 1/(fps_2-fps_1)
            fps_1 = fps_2

            cv2.putText(img,'TEST_POSE',(300,20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255), 2)
            cv2.putText(img,'FPS:',(580,20),cv2.FONT_HERSHEY_PLAIN,1,(209,0,178), 2)
            cv2.putText(img,str(int(fps)),(615,20),cv2.FONT_HERSHEY_PLAIN, 1,(209,0,178), 2)
            cv2.imshow("video", img)
            cv2.waitKey(1)
            pbar.set_description(f"Carregando")
            pbar.update(a)
            a = a+1
            
        else:
            print()
            pbar.close()
            break
    
            
if __name__ == "__main__":
    main()
"""