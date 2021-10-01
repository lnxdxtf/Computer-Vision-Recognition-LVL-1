import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, maxHands=2, detecionCon=0.5, trackCon=0.5): 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detecionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True ):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
 
        return img

    def findPosition (self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append(["id:",id,"x:",cx,"y:",cy])
                if draw:
                    cv2.circle(img,(cx,cy), 3, (0,255,0), cv2.FILLED)
                   
        return lmList

    
        
""" 


------------------------- CÓDIGO teste
---- BASTA importar da pasta e definindo como htm 
---- EXEMPLO: import my_modules.hand_tracking_module as htm
---- e cria teu objeto para chamar os metodos da classe 
----
----
------------------------------------------------------------
def main():
   
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        succes, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (580,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    

if __name__ == "__main__":
    main() 


"""