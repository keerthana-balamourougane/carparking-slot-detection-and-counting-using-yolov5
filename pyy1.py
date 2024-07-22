import cv2
import torch
import numpy as np
import time

points=[]
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)            
    
           


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('parking1.mp4')
#count=0
area=[(24,433),(9,516),(409,490),(786,419),(720,368)]
parking_slots = [
    [(40, 438),(17, 508), (85, 510), (102, 431)],  # Slot 1
    [(94, 432),(77, 509), (146, 509), (154, 426)],# Slot 2
    [(148,427),(140,509),(214,505),(214,424)],
    [(209,423),(207,503),(279,499),(270,420)],
    [(266,419),(277,494),(348,489),(329,419)],
    [(329,413),(347,492),(416,481),(390,409)],
    [(387,409),(416,481),(483,475),(448,403)],
    [(447,402),(480,473),(546,464),(506,400)], 
    [(503,397),(546,464),(606,454),(558,389)],
    [(558,387),(603,453),(662,443),(609,385)],
    [(608,381),(656,441),(704,433),(654,380)],
    [(651,377),(706,431),(745,421),(698,374)],
    # Add more slots as needed
]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,600))

    results=model(frame)
    for slot in parking_slots:
        slot_pts = np.array(slot, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [slot_pts], True, (0, 255, 0), 2)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if 'car' in d:
           results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
           if results>=0:
          # print(results)
             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
             cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
             list.append([cx])
           
    #cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    a=(len(list))
    cv2.putText(frame,str(a),(50,49),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
   
    time.sleep(0.1) 
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
