# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

model = tf.keras.models.load_model("C:/Users/shiva/OneDrive/Documents/python/C-110 std/keras_model.h5")
# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		#resize the frame
		image =cv2.resize(frame,(224,224))

		# expand the image
		t_img = np.array(image,dtype = np.float32)
		t_img = np.expand_dims(image,axis=0)

		# normalize it before feeding to the model
		normalize_img = t_img/255.0

		# get predictions from the model
		
		prediction = model.predict(normalize_img)
		print("prediction :",prediction)
		
		# Converting the data in the array to percentage confidence 
		rock = int(prediction[0][0]*100)
		paper = int(prediction[0][1]*100)
		scissor = int(prediction[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
