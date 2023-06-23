import cv2
import os

# Get the path to the folder containing the videos
video_folder = "D:\\Final year project\\data\\drive-download-20230519T073441Z-001\\Live"

# Create a new folder to store the snapshots
snapshot_folder = "D:\\Final year project\\data\\Live"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

# Iterate over all the videos in the folder
for video_file in os.listdir(video_folder):
    print(video_file)
    # Create a VideoCapture object for the video
    video = cv2.VideoCapture(os.path.join(video_folder, video_file))

    # Get the number of frames in the video
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Iterate over all the frames in the video
    for i in range(int(frame_count)):

        # Read the next frame from the video
        ret, frame = video.read()

        # If the frame was read successfully, save it to the snapshot folder
        if ret:
            if i%15==0:
                cv2.imwrite(os.path.join(snapshot_folder, "frame_{}_{}.jpg".format(video_file, i)), frame)

    # Release the VideoCapture object
    video.release()

