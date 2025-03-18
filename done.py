import cv2 as cv
import numpy as np

def lucas_kanade_optical_flow(prev_img, curr_img, points, window_size=15):
    Ix = cv.Sobel(prev_img, cv.CV_64F, 1, 0, ksize=3)
    Iy = cv.Sobel(prev_img, cv.CV_64F, 0, 1, ksize=3)
    It = curr_img.astype(np.float32) - prev_img.astype(np.float32)
    
    half_w = window_size // 2
    flow_vectors = np.zeros_like(points, dtype=np.float32)

    for i, pt in enumerate(points):
        x, y = int(pt[0, 0]), int(pt[0, 1])
        if x-half_w < 0 or y-half_w < 0 or x+half_w >= prev_img.shape[1] or y+half_w >= prev_img.shape[0]:
            continue  # Ignore out-of-bounds points
        
        Ix_win = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        Iy_win = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        It_win = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        
        A = np.vstack((Ix_win, Iy_win)).T
        b = -It_win.reshape(-1, 1)
        
        if A.shape[0] >= 2:  # Ensure sufficient points to solve
            nu = np.linalg.pinv(A) @ b
            flow_vectors[i] = nu.ravel()
    
    return flow_vectors

cap = cv.VideoCapture("videos/OPTICAL_FLOW.mp4")
ret, first_frame = cap.read()
if not ret:
    print("Error: Failed to load the video")
    exit()

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Recalculate good features to track in each frame
    prev_points = cv.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    if prev_points is None:
        prev_gray = gray.copy()
        continue  # Skip frame if no good features are found

    flow_vectors = lucas_kanade_optical_flow(prev_gray, gray, prev_points)
    good_new = prev_points + flow_vectors.reshape(-1, 1, 2)
    
    # Draw optical flow
    for i, (new, old) in enumerate(zip(good_new, prev_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        if 0 <= a < frame.shape[1] and 0 <= b < frame.shape[0]:  # Ensure points are within bounds
            color = (0, 255, 0)
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
            frame = cv.circle(frame, (int(a), int(b)), 3, color, -1)
    
    output = cv.add(frame, mask)
    cv.imshow("Lucas-Kanade Optical Flow", output)
    
    prev_gray = gray.copy()
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
