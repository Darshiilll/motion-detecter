import cv2
import datetime
import winsound

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(BASE_DIR, "alarm.wav")


cap = cv2.VideoCapture(0)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recording.avi', fourcc, 20.0, (640,480))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

alarm_played = False

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue

        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame1, "Motion Detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Timestamp
    cv2.putText(frame1,
                datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                (10, frame1.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Play alarm once per motion event
    if motion_detected:
        out.write(frame1)
        if not alarm_played:
            winsound.PlaySound(alarm_path, winsound.SND_FILENAME | winsound.SND_ASYNC)



            alarm_played = True
    else:
        alarm_played = False

    cv2.imshow("Motion Detector", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
