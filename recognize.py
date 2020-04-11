import cv2 as cv
"""Busca los archivos para reconocer cada parte"""
face_cascade = cv.CascadeClassifier('/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml')
font                   = cv.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
def detect(gray, frame): 
    """obtenemos una lista de todos los rostros encontrados en el frame"""
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        """Los ejes cartesianos se leen de arriba a abajo, por ende x = 0 esta arriba a la izquierda del frame encontrado"""
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        cv.putText(frame,'cabeza',(x+w,y),font,fontScale,fontColor,lineType)
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
        eyes = eye_cascade.detectMultiScale(roi_gray,2,20)

        for (sx, sy, sw, sh) in smiles: 
            cv.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (255,128,128), 2)
            cv.putText(frame,'sonrisa',(x-sx+w-sw,y-sy+h+sh),font,fontScale,fontColor,lineType)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color, (ex,ey),((ex+ew),(ey+eh)),(0,119,255),2)     
            cv.putText(frame,'ojos',(x-ex+w-ew,y-ey+h-3*eh),font,fontScale,fontColor,lineType)
    return frame 
"""El cap lo que hace es empezar a grabar"""    
cap = cv.VideoCapture(0)
while (True):
    valido,img = cap.read()
    if (valido):
        gray = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        canvas = detect(img, gray)
        cv.imshow('Reconocimiento facial',canvas)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break   
cap.release()
cv.destroyAllWindows()

