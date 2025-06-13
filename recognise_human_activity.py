# # Required imports
# from collections import deque
# import numpy as np
# import cv2y

# print(cv2.__version__)  # Fix version check syntax

# # Parameters class to store model paths and constants
# class Parameters:
#     def __init__(self):  # Fix the constructor method
#         self.CLASSES = open("model/action_recognition_kinetics.txt").read().strip().split("\n")
#         self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
#         self.VIDEO_PATH = None  # Set to None for webcam
#         self.SAMPLE_DURATION = 16  # Frames in the buffer
#         self.SAMPLE_SIZE = 112  # Model expects 112x112 input

# # Initialize parameters
# param = Parameters()

# # Check if model exists
# import os
# if not os.path.exists(param.ACTION_RESNET):
#     print("[ERROR] Model file not found! Please download resnet-34_kinetics.onnx")
#     exit()

# # Initialize a deque to store frames
# captures = deque(maxlen=param.SAMPLE_DURATION)

# # Load the human activity recognition model
# print("[INFO] Loading human activity recognition model...")
# net = cv2.dnn.readNet(param.ACTION_RESNET)

# # Open webcam or video file
# print("[INFO] Accessing video stream...")
# vs = cv2.VideoCapture(0 if param.VIDEO_PATH is None else param.VIDEO_PATH)

# if not vs.isOpened():
#     print("[ERROR] Could not open video stream!")
#     exit()

# while True:
#     # Read a frame from the webcam
#     grabbed, capture = vs.read()
    
#     if not grabbed:
#         print("[INFO] No capture read from stream - exiting")
#         break

#     # Resize frame and add to deque
#     capture = cv2.resize(capture, (550, 400))
#     captures.append(capture)

#     # Process only when enough frames are collected
#     if len(captures) < param.SAMPLE_DURATION:
#         continue

#     # Convert to a format the model expects
#     imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
#                                        (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
#                                        (114.7748, 107.7354, 99.4750),
#                                        swapRB=True, crop=True)

#     imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
#     imageBlob = np.expand_dims(imageBlob, axis=0)

#     # Pass through the model
#     net.setInput(imageBlob)
#     outputs = net.forward()
#     label = param.CLASSES[np.argmax(outputs)]  # Get the predicted action

#     # Display the prediction on the frame
#     cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
#     cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8, (0, 0, 0), 2)

#     # Show frame
#     cv2.imshow("Human Activity Recognition", capture)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Cleanup
# vs.release()
# cv2.destroyAllWindows()



#2ND CODE


# Required imports
from collections import deque
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

print(f"OpenCV Version: {cv2.__version__}")

# Parameters class to store model paths and constants
class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt").read().strip().split("\n")
        self.ACTION_RESNET = "model/resnet-34_kinetics.onnx"
        self.VIDEO_PATH = None  # Default: None (uses webcam)
        self.SAMPLE_DURATION = 16  # Number of frames to collect for prediction
        self.SAMPLE_SIZE = 112  # Model input size (112x112)

# Function to let user select a video file
def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide main Tkinter window
    file_path = filedialog.askopenfilename(title="Select Video File",
                                           filetypes=[("MP4 Files", "*.mp4"), 
                                                      ("AVI Files", "*.avi"),
                                                      ("MOV Files", "*.mov"),
                                                      ("All Files", "*.*")])
    return file_path if file_path else None  # Return file path or None

# Initialize parameters
param = Parameters()

# Ask user if they want to upload a video
use_uploaded_video = input("Do you want to upload a video? (y/n): ").strip().lower()
if use_uploaded_video == 'y':
    param.VIDEO_PATH = select_video()
    if not param.VIDEO_PATH:
        print("[ERROR] No video file selected! Exiting...")
        exit()

# Check if model file exists
if not os.path.exists(param.ACTION_RESNET):
    print("[ERROR] Model file not found! Please download resnet-34_kinetics.onnx")
    exit()

# Load the action recognition model
print("[INFO] Loading human activity recognition model...")
net = cv2.dnn.readNet(param.ACTION_RESNET)

# Open webcam or video file
video_source = 0 if param.VIDEO_PATH is None else param.VIDEO_PATH
print(f"[INFO] Accessing video stream from: {'Webcam' if video_source == 0 else param.VIDEO_PATH}")
vs = cv2.VideoCapture(video_source)

if not vs.isOpened():
    print("[ERROR] Could not open video stream!")
    exit()

# Initialize a deque to store frames
frames_buffer = deque(maxlen=param.SAMPLE_DURATION)

while True:
    # Read a frame from the video stream
    grabbed, frame = vs.read()
    
    if not grabbed:
        print("[INFO] No frame read from stream - exiting")
        break

    # Resize and normalize frame
    frame_resized = cv2.resize(frame, (550, 400))
    frames_buffer.append(frame_resized)

    # Process only when enough frames are collected
    if len(frames_buffer) < param.SAMPLE_DURATION:
        continue

    # Convert frames to a format suitable for the model
    blob = cv2.dnn.blobFromImages(frames_buffer, 1.0,
                                  (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                  (114.7748, 107.7354, 99.4750),
                                  swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # Run the model
    net.setInput(blob)
    outputs = net.forward()
    predicted_action = param.CLASSES[np.argmax(outputs)]  # Get predicted action label

    # Display the action prediction on the frame
    cv2.rectangle(frame_resized, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(frame_resized, predicted_action, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2)

    # Attempt to display using OpenCV
    try:
        cv2.imshow("Human Activity Recognition", frame_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except cv2.error as e:
        print("[WARNING] OpenCV GUI support is missing! Using Matplotlib for display.")
        
        # Display using Matplotlib as a fallback
        plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

# Cleanup
vs.release()
cv2.destroyAllWindows()
