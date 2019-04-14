# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse

from imutils.video import VideoStream
from collections import deque
import numpy as np
import imutils
import cv2
import time

# TEST VARS
NON_MAX_SUPRESSION = True

# Keep frame W and H
(W, H) = (None, None)

# Video capture
vs = VideoStream(src=3).start()
# Warm up camera
time.sleep(2.0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	frame = vs.read()
	frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    # DEBUG: Checking camera crash
    # if frame is None:
    #    print("Missed frame")
    # if W is None or H is None:
	#	(H, W) = frame.shape[:2]
    # detect people in the image
	start = datetime.datetime.now()
	(rects, weights) = hog.detectMultiScale(
		frame, 
		winStride=(4, 4),
		padding=(8, 8), 
		scale=1.05 #Needs to be tuned to find best performance  
	)
	print("[INFO] detection took: {}s".format((datetime.datetime.now() - start).total_seconds()))

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	if NON_MAX_SUPRESSION is True:
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	else:
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	
	#Close sript
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()    