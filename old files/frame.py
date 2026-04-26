import cv2

cap = cv2.VideoCapture("haha.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Actual video frame width: {frame_width} pixels")
print(f"Actual video frame height: {frame_height} pixels")

cap.release()