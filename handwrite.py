import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# drawing points
points = [deque(maxlen=1024)]
index = 0

# colors
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

current_color = BLUE

# canvas
canvas = np.ones((480,640,3),dtype="uint8")*255

# mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# camera
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame,1)

    # draw buttons
    cv2.rectangle(frame,(10,10),(110,60),(0,0,0),2)
    cv2.rectangle(frame,(130,10),(230,60),BLUE,2)
    cv2.rectangle(frame,(250,10),(350,60),GREEN,2)
    cv2.rectangle(frame,(370,10),(470,60),RED,2)

    cv2.putText(frame,"CLEAR",(20,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
    cv2.putText(frame,"BLUE",(145,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,BLUE,2)
    cv2.putText(frame,"GREEN",(260,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,GREEN,2)
    cv2.putText(frame,"RED",(395,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,RED,2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for handLms in result.multi_hand_landmarks:

            h,w,c = frame.shape

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            cv2.circle(frame,(x,y),8,(0,0,0),-1)

            # check button click
            if y <= 60:

                # clear
                if 10 <= x <= 110:
                    canvas[:] = 255
                    points = [deque(maxlen=1024)]
                    index = 0

                # blue
                elif 130 <= x <= 230:
                    current_color = BLUE

                # green
                elif 250 <= x <= 350:
                    current_color = GREEN

                # red
                elif 370 <= x <= 470:
                    current_color = RED

            else:

                points[index].appendleft((x,y))

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # draw lines
    for i in range(len(points)):
        for j in range(1,len(points[i])):

            if points[i][j-1] is None or points[i][j] is None:
                continue

            cv2.line(frame, points[i][j-1], points[i][j], current_color,5)
            cv2.line(canvas, points[i][j-1], points[i][j], current_color,5)

    cv2.imshow("Camera", frame)
    cv2.imshow("Paint", canvas)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()