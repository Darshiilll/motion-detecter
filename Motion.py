import cv2
import datetime
import os
import time
import winsound

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(VIDEO_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

# real camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

recording = False
out = None
last_motion_time = 0
STOP_DELAY = 2

prev_brightness = 0

# delete videos older than 7 days
def cleanup_old_videos():
    now = time.time()
    for file in os.listdir(VIDEO_DIR):
        path = os.path.join(VIDEO_DIR, file)
        if os.path.isfile(path):
            if now - os.path.getmtime(path) > 7 * 24 * 60 * 60:
                os.remove(path)

cleanup_old_videos()

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    brightness = gray.mean()

    # night sensitivity
    threshold_value = 15 if brightness < 60 else 25

    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    moving_objects = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # ignore tiny noise
        if area < 2500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # ignore thin shapes (fan blades, shadows)
        if w < 40 or h < 40:
            continue

        moving_objects += 1
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)

    # require multiple blobs (real person)
    if moving_objects >= 2:
        motion_detected = True

    # ignore sudden lighting change
    if abs(brightness - prev_brightness) > 25:
        motion_detected = False
    prev_brightness = brightness

    # timestamp
    cv2.putText(frame1,
        datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        (10, frame1.shape[0]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ---- RECORDING LOGIC ----
    if motion_detected:
        last_motion_time = time.time()

        if not recording:
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.avi")
            filepath = os.path.join(VIDEO_DIR, filename)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))

            recording = True
            winsound.MessageBeep(winsound.MB_ICONHAND)

    if recording and out:
        out.write(frame1)

        if time.time() - last_motion_time > STOP_DELAY:
            recording = False
            out.release()
            out = None

    # -------------------------

    cv2.imshow("Motion Detector", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
