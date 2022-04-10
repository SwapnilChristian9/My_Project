# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import Adafruit_CharLCD as LCD
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

#for GPIO labeling
GPIO.setmode(GPIO.BCM)

#for LCD Interface
lcd1 = 12
lcd2 = 7
lcd3 = 8
lcd4 = 25
lcd5 = 24
lcd6 = 23
lcd = LCD.Adafruit_CharLCD(lcd1, lcd2,lcd3, lcd4, lcd5, lcd6, 0, 16, 2)
lcd.clear()

#for servo Interface
GPIO.setup(17,GPIO.OUT)
p = GPIO.PWM(17,50)
p.start(0)

#LED Interface
GPIO.setup(27,GPIO.OUT)#GREEN LED
GPIO.setup(22,GPIO.OUT)#RED LED

#BUZZER Interface
GPIO.setup(4,GPIO.OUT) #BUZZER 

#function1 call when mask is detected
def function1():
        GPIO.output(27, True) #green led on
        print("mask detected") 
        lcd.message("Mask detected")
        print("please check your temperature")
        lcd.message("please check\nyour temperature")
        #delay time.sleep(time in second)
        time.sleep(3)
        #check temperature
        bus = SMBus(1)
        sensor = MLX90614(bus, address=0x5A)
        print "Ambient Temperature :", sensor.get_ambient()
        temp_in_celsius = sensor.get_object_1()
        temp = (temp_in_celsius * 9/5) + 32
        print "Object Temperature :", temp
        bus.close()
        #temp = 101
        
        print("Your Temperaure:",temp)
        lcd.message("Your Temperaure:\n",temp)
        
        if (temp > 100):
                GPIO.output(27, False) #green led off
                GPIO.output(22, True)  #red led on
                print("Your Temperature is HIGH")
                lcd.message("Your Temperature\nis HIGH")
                print("BUZZER ON")
                GPIO.output(4, True)
                #delay for 3 seconds
                time.sleep(3)
                print("BUZZER OFF")
                GPIO.output(4, False)
                GPIO.output(22, False) #red led off
        else:
                lcd.message("Your Temperature\nis Normal")
                time.sleep(0.5)
                print("WELCOME")
                lcd.message("WELCOME")
                #Door open for 10 seconds (servo operation)
                p.ChangeDutyCycle(7.5)
                time.sleep(10)
                p.ChangeDutyCycle(0)
                #Door closed
                GPIO.output(27, False) #green led off
        lcd.clear()
        
#function2 call when mask is not detected
def function2():
        GPIO.output(22, True) #red led on
        print("mask not detected")
        lcd.message("Mask not\ndetected")
        print("BUZZER ON")
        GPIO.output(4, True)
        print("Please wear your mask")
        lcd.message("Please wear\nyour mask")
        time.sleep(3)
        print("BUZZER OFF")
        GPIO.output(4, False)
        GPIO.output(22, False) #red led off
        lcd.clear()
        
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"D:\SEM-8\Final_Project\Face-Mask-Detection\face_detector_model\deploy.prototxt"
weightsPath = r"D:\SEM-8\Final_Project\Face-Mask-Detection\face_detector_model\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		if(mask > withoutMask):
			label = "Mask"
			print("mask")
			color = (0, 255, 0)
			function1()
		else:
			label ="No Mask",
			print("no mask")
			color = (0, 0, 255)
			function2()


		#label = "Mask" if mask > withoutMask else "No Mask"
		#color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
