import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                         min_detection_confidence=self.detectionCon,
                                         min_tracking_confidence=self.trackCon)
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            
            for handLms in self.results.multi_hand_landmarks:  # for multiple hands
                if draw:
                    # hand connections make lines b/w points like edges in graph 
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPositions(self, img, handNo=0, draw=True, positions=[]):

        lmList = [] 
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # x and y are ratio of image 
                lmList.append([id, cx, cy]) # adding positions of landmarks to list

            if draw:
                for pos in positions:
                    cv2.circle(img, (lmList[pos][1], lmList[pos][2]), 15, (255, 0, 0), cv2.FILLED)

        return lmList



def main():
    prevTime = 0
    curTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()
    
    while True:
        success, img = cap.read()

        img = detector.findHands(img, draw=True)
        # find positions of landmarks in image
        lmList = detector.findPositions(img, draw=True, positions=[4, 8, 12, 16, 20])
        if len(lmList) != 0:
            id, x, y = lmList[4]

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

