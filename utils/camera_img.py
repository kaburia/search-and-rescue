from picamera2 import Picamera2
from time import sleep
import os
from datetime import datetime

from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# Get current time as string
def get_current_time():
    time_now = datetime.now()
    current_time = time_now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

# Initialize camera

def init_camera(mode="still"):
    camera = Picamera2()

    if mode == "still":
        # Configure for full 12MP still capture
        config = camera.create_still_configuration(
            main={"size": (4056, 3040)}  # 12MP
        )
        camera.configure(config)
        camera.start()
         # Warm-up before capture
        return camera

    elif mode == "video":
        # Configure for 1080p ~30fps video
        config = camera.create_video_configuration(
            main={"size": (1920, 1080)},
            controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps
        )
        camera.configure(config)
        print("Camera initialized in video mode (use capture_video to record).")
        return camera

    else:
        raise ValueError("Mode must be either 'still' or 'video'")


# Take picture
def capture_image(todays_date):
    current_time = get_current_time()

    # Ensure folder exists
    if not os.path.exists('deploy-test-data'):
        os.makedirs('deploy-test-data', exist_ok=True)

    image_name = f"deploy-test-data/{todays_date}/image_{current_time.replace(' ', '_').replace(':', '-')}.jpg"

    # Open camera
    camera = init_camera()
    sleep(2) # Camera warm-up time

    # Capture image
    camera.capture_file(image_name)
    print(f"Image captured and saved as {image_name}")

    # Close camera
    camera.close()

    return image_name


# Record video
import subprocess

# Record video
def capture_video(todays_date, duration=10, resolution=(1920, 1080), fps=30, fmt="h264"):
    current_time = get_current_time()

    # Ensure folder exists
    folder = f"deploy-test-data-video/{todays_date}"
    os.makedirs(folder, exist_ok=True)

    # Base filename
    base_name = f"{folder}/video_{current_time.replace(' ', '_').replace(':', '-')}"
    video_name = f"{base_name}.h264"

    # Open camera via init_camera
    camera = init_camera(mode="video")

    # Reconfigure to custom resolution & fps if needed
    config = camera.create_video_configuration(
        main={"size": resolution},
        controls={"FrameDurationLimits": (int(1e6/fps), int(1e6/fps))}
    )
    camera.configure(config)

    # Encoder & output
    encoder = H264Encoder(bitrate=10_000_000)
    output = FileOutput(video_name)

    # Record
    camera.start_recording(encoder, output)
    print(f"Recording video for {duration}s at {resolution[0]}x{resolution[1]} @ {fps}fps → {video_name}")
    sleep(duration)

    # Stop recording & cleanup
    camera.stop_recording()
    camera.close()

    # Convert to mp4 if requested
    if fmt == "mp4":
        mp4_name = f"{base_name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_name, 
            "-c", "copy", mp4_name
        ], check=True)
        os.remove(video_name)  # delete raw h264 if not needed
        print(f"Video converted and saved as {mp4_name}")
        return mp4_name

    print(f"Video saved as {video_name}")
    return video_name
