import subprocess
import time
import cv2
import numpy as np

# Image dimensions
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2

# Command to launch libcamera/rpicam feed
rpicam_cmd = [
    "rpicam-vid",
    "-t", "0",
    "-n",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--codec", "yuv420",
    "--framerate", "30",
    "-o", "-"
]

# Start subprocess
process = subprocess.Popen(
    rpicam_cmd, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.DEVNULL, 
    bufsize=FRAME_SIZE * 2
)

prev_time = time.time()

try:
    while True:
        # Read exact frame size from stdout
        raw_image = process.stdout.read(FRAME_SIZE)
        if len(raw_image) != FRAME_SIZE:
            print("Camera stream disconnected.")
            break

        # Convert raw YUV420 buffer to BGR frame
        yuv_data = np.frombuffer(raw_image, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Display FPS on top-left corner
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # Show live view
        cv2.imshow("Raspberry Pi - Real-Time Camera View", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    process.terminate()
    process.wait()
    cv2.destroyAllWindows()