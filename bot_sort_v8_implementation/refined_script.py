import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import yt_dlp

def predict_trajectory(track, future_steps=30):
    """
    Project future trajectory based on current velocity.
    Uses the last few points to estimate stable velocity.
    """
    if len(track) < 5:
        return [] # Not enough history to project a stable trajectory

    # Use the last point and a point from slightly in the past to estimate velocity
    past_pt = np.array(track[-5])
    current_pt = np.array(track[-1])
    
    # Calculate velocity (change in position per frame)
    velocity = (current_pt - past_pt) / 5.0
    
    # Project future points
    future_track = []
    for step in range(1, future_steps + 1):
        future_pt = current_pt + velocity * step
        future_track.append((int(future_pt[0]), int(future_pt[1])))
    
    return future_track

def main():
    # Load standard pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # ==========================================
    #             YOUTUBE LIVE STREAM
    # ==========================================
    youtube_url = "https://www.youtube.com/watch?v=iaBfYxbmwXA"
    print(f"Connecting to YouTube stream at {youtube_url} (this may take a few seconds)...")
    
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info.get('url')

    if not stream_url:
        print("Error: Could not extract stream URL from YouTube link.")
        return
        
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open YouTube video stream. The stream may be offline.")
        return

    # Store the track history
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Stream ended or failed to read frame. Exting loop.")
            break

        # Run YOLO tracking on the current frame
        # stream=False is idiomatic for frame-by-frame loops in Ultralytics.
        # Use a standard tracker like 'botsort.yaml' or 'bytetrack.yaml' for best results.
        results = model.track(frame, persist=True, tracker=r"c:\Users\MVMaas\.antigravity\bot_sort_v8_implementation\tester.yaml", conf=0.15,verbose=False)

        # There is only one frame passed, so we access results[0]
        r = results[0]
        annotated_frame = r.plot() # Draw the standard YOLO boxes and labels
        
        # Check if any objects were tracked
        if r.boxes.id is not None:
            boxes = r.boxes.xywh.cpu()
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (float(x), float(y))
                
                # Update tracking history
                track = track_history[track_id]
                track.append(center)
                if len(track) > 30:
                    track.pop(0) # Keep history bounded

                # 1. Draw the historical path (breadcrumb trail)
                if len(track) > 1:
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(235, 219, 11), thickness=2)

                # 2. Project and draw the future trajectory
                future_path = predict_trajectory(track, future_steps=20)
                if future_path:
                    future_points = np.array(future_path).astype(np.int32).reshape((-1, 1, 2))
                    # Draw projected path (dotted or distinct color)
                    cv2.polylines(annotated_frame, [future_points], isClosed=False, color=(0, 165, 255), thickness=2)

        cv2.imshow("YOLOv8 Object Tracking & Projection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
