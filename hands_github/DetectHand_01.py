import cv2
import mediapipe as mp
import time
import my_modules.hand_tracking_module as htm

#variaveis para calcular o fps
fps_1_Time = 0
fps_2_time = 0

cap = cv2.VideoCapture(0) # define a camera
#definindo detectorR
detector = htm.handDetector(maxHands=2) #(self, mode=False, maxHands=2, detecionCon=0.5, trackCon=0.5) maxHands: maximo de mao que podem ser detectadas ; detectionCon: precisao de detectao; trackCon: precisao de rastreamento 

while True:
    succes, img = cap.read()
    img = detector.findHands(img, draw=True) #draw desenha e liga os circulos padroes
    list_maopts = detector.findPosition(img, draw=True) #draw - desenha/define os circulos estilizados !!!!!!Apenas a mão que esta sendo detectada e rastreada!!!!! APENAS UMA

    """ if len(list_maopts) != 0:
        print(list_maopts[0]) """ #devolve a posição de determinado ponto da mão - onde o 0 é o inicio da mão de acordo com o arquivo hand_pts.png

    fps_2_time = time.time()
    fps = 1/(fps_2_time - fps_1_Time)
    fps_1_Time = fps_2_time
    
    cv2.putText(img, str(int(fps)), (580,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3) # colocando o fps na tela
    cv2.imshow('Mãos_detector', img) # abrindo a tela

    cv2.waitKey(1)