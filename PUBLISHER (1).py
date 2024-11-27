
#importer les libraries
import paho.mqtt.publish as publish
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

buzzer_pin = 29
GPIO.setup(buzzer_pin, GPIO.OUT)


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

#fct  etat conducteur ( yawning sleeping)
    while alarm_status:
        print('DRIVER IS SLEEPING')
        publish.single("iseoc2021/test", "DRIVER IS SLEEPING", hostname="test.mosquitto.org")
       
    if alarm_status2:
        print('DRIVER IS YAWNING')
        
        publish.single("iseoc2021/topic", "DRIVER IS YAWNING", hostname="test.mosquitto.org")
        saying = True
        
        saying = False


#calcul eye ratio 
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) #distance euclidienne 
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)  # eye ratio

    return ear

def final_ear(shape):
    #init index eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#extraire coordonnées yeux
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    #calcul ratio
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)
# mouth 
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# choix webcam
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

#parametres
EYE_AR_THRESH = 0.27 # min eye ratio
EYE_AR_CONSEC_FRAMES = 5
YAWN_THRESH = 20 #min mouth rtio
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #detecter visage
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  #land marks (eye,lips)

#lancer video
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

while True:

    frame = vs.read() #lecture frame
    frame = imutils.resize(frame, width=450) #resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir frame en grayscale

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

   
    for (x, y, w, h) in rects:
        #construire dlibrectangle a partir de cascade de Har
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect) #determiner landmarks
        shape = face_utils.shape_to_np(shape) # converit landmarks en tableau numpy

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)
        # contours bouche et yeux
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        #detecter drowsiness
        #si ratio < duree blink on lance le compteur
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            #si compteur >  nb frames alors chauffeur endormi
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
					GPIO.output(buzzer_pin, GPIO.HIGH)

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False
        # detection yawn
        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
					GPIO.output(buzzer_pin, GPIO.HIGH)
        
        else:
            alarm_status2 = False
            
    #ajout texte sur la frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break




 
cv2.destroyAllWindows()
vs.stop()

