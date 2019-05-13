# Computer-Vision-Based-Smart-Selfie
In this project, one can learn to develop a computer vision based smart selfie application which can take snaps/pictures automatically when you smile using the facial feature recognition algorithm and save it on your device.
import time
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 6) 
    
    for (x, y, w, h) in faces: 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
            return frame
        
cap = cv2.VideoCapture(0) 
count=0
start=0

while True: 
    _, frame = cap.read()  
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
      
    canvas = detect(gray, frame)
    
    if canvas is not None:
        #cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        count+=1
        
    if(count==5):
        start=time.time()
    if(start!=0 and time.time()-start<3):
        cv2.putText(frame,str(int(time.time()-start)+1),(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    if(count>5 and time.time()-start>3):
        start=0
        count=0
        cv2.imwrite('smile.jpg',frame)
        
    cv2.imshow('Video', frame)  
  
    if cv2.waitKey(1) ==13:                
        break 
cap.release()                                  
cv2.destroyAllWindows() 
