# Maybe needed
from __future__ import print_function
from imutils import paths
import argparse
import datetime

# Needed
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
from collections import deque
import numpy as np
import imutils
import cv2
import time
import json
import subprocess
import base64

ap = argparse.ArgumentParser()
# TEST PARAM:
# construct the argument parse and parse the arguments
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-w", "--win-stride", type=str, default="(2, 2)",
	help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(4, 4)",
	help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.4,
	help="image pyramid scale")
ap.add_argument("-o", "--overlap", type=float, default=0.05,
	help="overlap threshold")
args = vars(ap.parse_args())
winStride = eval(args["win_stride"])
padding = eval(args["padding"])

# Aux Variables
N = 1  # Update frame each N seconds
VIDEO_STREAM = True
# Tuple
GoodFrames = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
LessBlurred = 0
BestFrame = 0
BestFrameID = 0


# Packet INFO
Pi_ID = 0
Frame_number = 0

# Video capture
# Warm up camera
if VIDEO_STREAM:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# initialize the video stream, pointer to output video file, and
# frame dimensions
else:
	vs = cv2.VideoCapture(args["input"])

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
	for x in range(0, 24*N):
		if VIDEO_STREAM:
			frame = vs.read()
		else:
			(grabbed, frame) = vs.read()
		bluriness = cv2.Laplacian(frame, cv2.CV_64F).var()
		if bluriness > LessBlurred:
			BestFrame = frame
			BestFrameID = x
			LessBlurred = bluriness
	LessBlurred = 0
	BestFrame = imutils.resize(BestFrame, width=min(400, BestFrame.shape[1]))
	(rects, weights) = hog.detectMultiScale(
		BestFrame,
		winStride=winStride,		# OLD (4,4)
		padding=padding,    # OLD (8,8)
		scale=args["scale"]  	# OLD 1.05	# Needs to be tuned to find best performance
	)
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=args["overlap"])
	for (xA, yA, xB, yB) in pick:
		# .item converts numpy int64 dtype to python int type
		packet = {
			"Pi_ID": Pi_ID,
			"Frame_number": BestFrameID,
			"X": xA.item(),
			"Y": yA.item(),
			"W": (xB-xA).item(),
			"H": (yB-yA).item(),
		}
		JSON_PACKET = json.dumps(packet)
		print(JSON_PACKET)
		encoded = base64.b64encode(JSON_PACKET.encode('utf-8'))
		subprocess.Popen(["bash", "process.sh", encoded])
		cv2.rectangle(BestFrame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	cv2.imshow("Frame", BestFrame)
	cv2.waitKey(N*1200)
	print(BestFrameID)