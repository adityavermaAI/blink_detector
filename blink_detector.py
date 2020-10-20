import cv2
import dlib
import numpy as np
from math import hypot

def midpoint(p1,p2):
    return (int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))

detector =dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('blinking_eyes_face_2x.mp4')

while True:
    flag, cam_image = cap.read()
    cam_image_gray = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)

    faces = detector(cam_image_gray)
    for face in faces:
        print(face)
        landmarks = predictor(cam_image_gray, face)

        ll = (landmarks.part(36).x, landmarks.part(36).y)
        lr = (landmarks.part(39).x, landmarks.part(39).y)
        lct = midpoint(landmarks.part(37), landmarks.part(38))
        lcb = midpoint(landmarks.part(40), landmarks.part(41))

        cv2.line(cam_image, ll, lr, (0,255,0), 2)
        cv2.line(cam_image, lct, lcb, (0,255,0), 2)

        ver_line_len = hypot(lct[0]-lcb[0], lct[1]-lcb[1])
        hor_line_len = hypot(ll[0]-lcb[0], lr[1]-lcb[1])

        l_ratio = hor_line_len/ver_line_len

        rl = (landmarks.part(42).x, landmarks.part(42).y)
        rr = (landmarks.part(45).x, landmarks.part(45).y)
        rct = midpoint(landmarks.part(43), landmarks.part(44))
        rcb = midpoint(landmarks.part(46), landmarks.part(47))

        cv2.line(cam_image, rl, rr, (0,255,0), 2)
        cv2.line(cam_image, rct, rcb, (0,255,0), 2)

        ver_line_len = hypot(rct[0]-rcb[0], rct[1]-rcb[1])
        hor_line_len = hypot(rl[0]-rcb[0], rr[1]-rcb[1])

        r_ratio = hor_line_len/ver_line_len

        blinking_ratio = l_ratio + r_ratio
        print(blinking_ratio)

        if blinking_ratio>4:
            cv2.putText(cam_image, 'BLINKING', (50,100), cv2.FONT_HERSHEY_PLAIN, 8, (0,0,255), 5)
    
    cv2.imshow('video', cam_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()