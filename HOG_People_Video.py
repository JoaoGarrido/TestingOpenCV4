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
import urllib
import os

# TEST VARS
DEBUG = False
DEBUG_show = True
DEBUG_FPS = True
OPT_JSON_WAY = False
# True->Single Object detected JSON File
# False ->Single Frame JSON File
NON_MAX_SUPRESSION = True

# Packet INFO
Pi_ID = 0
Frame_number = 0

# Keep frame W and H
(W, H) = (None, None)

# Local Video Capture
# vs = VideoStream(src=0).
# Warm up camera
# time.sleep(2.0)

# Receving videoStream
camIP = "http://192.168.103.108:8000/stream.mjpg"
# stream = cv2.VideoCapture(camIP)
stream = urllib.urlopen(cam2)
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# TEST PARAM:
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
	help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
	help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
	help="image pyramid scale")
ap.add_argument("-o", "--overlap", type=float, default=0.55,
	help="overlap threshold")
args = vars(ap.parse_args())
winStride = eval(args["win_stride"])
padding = eval(args["padding"])

while True:

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	
	# VideoStreaming LOCAL PROCESSING
	#frame = vs.read()
	# VideoStreaming Server PROCESSING
	# retval, frame = stream.read()
	bytes+=stream.read(1024)
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')
	if a!=-1 and b!=-1:
		jpg = bytes[a:b+2]
		bytes= bytes[b+2:]
	frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
	# we now have frame stored in frame.
	Frame_number = Frame_number+1 
	frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	
	# DEBUG: Checking camera crash	
	if frame is None and DEBUG:
		print("Missed frame")
	# if W is None or H is None:
	#	(H, W) = frame.shape[:2]
	# detect people in the image
	if DEBUG_FPS is True:
		start = datetime.datetime.now()
	
	# HOG
	# _____________________________/___________________________
	(rects, weights) = hog.detectMultiScale(
		frame,
		winStride=winStride,		# OLD (4,4)
		padding=padding,	#OLD (8,8)
		scale=args["scale"]  	#	OLD 1.05	# Needs to be tuned to find best performance
	)
	if DEBUG_FPS is True:
		print("[INFO] detection took: {}s".format(
			(datetime.datetime.now() - start).total_seconds()))
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	#if NON_MAX_SUPRESSION is True:
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=args["overlap"])
	for (xA, yA, xB, yB) in pick:
		# .item converts numpy int64 dtype to python int type
		packet = {
			"Pi_ID": Pi_ID,
			"Frame_number": Frame_number,
			#"Timestamp": datetime.datetime.now().item(),
			"X": xA.item(),
			"Y": yA.item(),
			"W": (xB-xA).item(),
			"H": (yB-yA).item(),
		}
		JSON_PACKET = json.dumps(packet)
		encoded = base64.b64encode(JSON_PACKET.encode('utf-8'))
		subprocess.Popen(["bash", "process.sh", encoded])
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	# _____________________________/___________________________
	#else:
	#	for (x, y, w, h) in rects:
	#		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# DEBUG - show()
	if DEBUG_show is True:
		cv2.imshow("Frame", frame)
	# Close sript
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()    