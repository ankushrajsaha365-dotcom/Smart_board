import cv2
import numpy as np

# --- Parameters ---
pen_colors = [(0,0,255), (0,255,0), (255,0,0)]  # Red, Green, Blue
color_names = ['Red', 'Green', 'Blue']
pen_index = 0
pen_thickness = 5
eraser_thickness = 50
prev_point = None
canvas = None
mode = 'draw'  # 'draw' or 'erase'

print("Keys: r/g/b = change color, e = erase, d = draw, c = clear, q = quit")

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

def get_topmost(contour):
    return tuple(contour[contour[:,:,1].argmin()][0])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # --- Background subtraction to detect moving object (hand) ---
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # --- Find contours ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 3000:
            cx, cy = get_topmost(c)
            cv2.circle(frame, (cx, cy), 10, (0,255,0), -1)

            # --- Gesture-based mode ---
            if mode != 'erase':  # draw only if not in erase mode
                mode = 'draw'

            # --- Smooth fingertip movement ---
            if prev_point is None:
                prev_point = (cx, cy)
            else:
                cx = int(0.7*prev_point[0] + 0.3*cx)
                cy = int(0.7*prev_point[1] + 0.3*cy)

            # --- Draw or erase ---
            if mode == 'draw':
                cv2.line(canvas, prev_point, (cx, cy), pen_colors[pen_index], pen_thickness)
            elif mode == 'erase':
                cv2.line(canvas, prev_point, (cx, cy), (0,0,0), eraser_thickness)

            prev_point = (cx, cy)
        else:
            prev_point = None
    else:
        prev_point = None

    # --- Overlay canvas ---
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask_canvas)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    frame_fg = cv2.bitwise_and(canvas, canvas, mask=mask_canvas)
    output = cv2.add(frame_bg, frame_fg)

    cv2.putText(output, f"Mode: {mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if mode=='draw':
        cv2.putText(output, f"Color: {color_names[pen_index]}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, pen_colors[pen_index], 2)

    cv2.imshow("Hand Smart Board", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('e'):
        mode = 'erase'
    elif key == ord('d'):
        mode = 'draw'
    elif key == ord('r'):
        pen_index = 0
    elif key == ord('g'):
        pen_index = 1
    elif key == ord('b'):
        pen_index = 2

cap.release()
cv2.destroyAllWindows()
