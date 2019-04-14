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

# Video capture
vs = VideoStream(src=2).start()
# Warm up camera
time.sleep(2.0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#TEST PARAM:
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
	BIG_JSON = ''
	frame = vs.read()
	Frame_number = Frame_number+1 
	frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	
	# DEBUG: Checking camera crash
	if frame is None and DEBUG:
		print("Missed frame")
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# detect people in the image
	if DEBUG_FPS is True:
		start = datetime.datetime.now()
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
	if NON_MAX_SUPRESSION is True:
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
	else:
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	if OPT_JSON_WAY is False:
		with open('packet.json', 'w') as outfile:
			outfile.write("{ %s }" % BIG_JSON)
	# DEBUG - show()
	if DEBUG_show is True:
		cv2.imshow("Frame", frame)
	
	# Close sript
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()    