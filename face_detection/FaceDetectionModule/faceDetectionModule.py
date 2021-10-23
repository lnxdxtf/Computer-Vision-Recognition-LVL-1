import cv2
import mediapipe as mp
import time



#CLASS FaceDetector

class FaceDetector():
    def __init__(self, minDetectionCon= 0.5, ):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self, img, draw= True, info= True, showScore= False ):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.FaceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                imgH, imgW, imgC = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * imgW), int(bboxC.ymin * imgH), \
                    int(bboxC.width * imgW), int(bboxC.height * imgH)
        
                bboxs.append([id, bbox, detection.score])
                x, y, w, h = bbox
                x1, y1 = x + w, y + h 
                if info:
                    img = self.extraDraw(img,bbox)
                    cv2.putText(img, f"ID: {id}", (bbox[0],
                    bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,59), 1)
                    if showScore:
                        cv2.putText(img, f"Con: {int(detection.score[0] * 100)}%", (x1-80,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,59), 1)

        return img, bboxs
    
    def extraDraw(self, img, bbox, l=35, t = 2):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h 

        cv2.rectangle(img,bbox, (0,0,255), t)
        #top rect
        cv2.line(img, (x,y),(x1, y),(0,0,255), t)
        cv2.line(img, (x,y),(x, y - l),(0,0,255), t)
        cv2.line(img, (x,y-l),(x1, y-l), (0,0,255), t)
        cv2.line(img, (x1, y-l),(x1,y), (0,0,255), t )
        
        return img

##/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
def main():
    v1 = "./Videos/6.mp4"
    v0 = 0
    cap = cv2.VideoCapture(v1)
    pTime = 0
    #creating object
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img , bboxs = detector.findFaces(img)

        cTime = time.time() #cTime = current time 
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(600,30),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 2)
        cv2.imshow("face_detect_img",img)
        cv2.waitKey(20) 


if __name__ == "__main__":
    main()

