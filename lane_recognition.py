import cv2
import numpy as np
import lanes

lane_recognition = cv2.VideoCapture('Driving.mp4')

if not lane_recognition.isOpened():
    print('Error - видео не открывается или отсутсвует в корне проекта')

cv2.waitKey(1)

while lane_recognition.isOpened():
    _, frame = lane_recognition.read()
    cv2.resizeWindow('lane_recognition', 900, 700)
    copy_img = np.copy(frame)

    try:
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        # убираем шумы и обьединяем линии
        lines = cv2.HoughLinesP(frame, 2, np.pi / 180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        averaged_lines = lanes.average_slope_intercept(frame, lines)
        line_image = lanes.display_lines(copy_img, averaged_lines)

        combo = cv2.addWeighted(copy_img, 0.8, line_image, 0.5, 1)
        cv2.imshow('lane_recognition', combo)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        lane_recognition.release()
        cv2.destroyAllWindows()


lane_recognition.release()
cv2.destroyAllWindows()
