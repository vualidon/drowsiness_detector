import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import numpy as np
import cv2
import subprocess
import threading
import playsound

def playsounds():
	playsound.playsound('mixkit-dog-barking-twice-1.wav')

def draw_landmarks(image, outputs, land_mark, color):
	height, width = image.shape[:2]

	for face in land_mark:
		point = outputs.multi_face_landmarks[0].landmark[face]

		point_scale = ((int)(point.x * width), (int)(point.y * height))

		cv.circle(image, point_scale, 2, color, 1)


def euclidean_distance(image, top, bottom):
	height, width = image.shape[:2]

	point1 = top.x * width, top.y * height
	point2 = bottom.x * width, bottom.y * height

	return dis.euclidean(point1, point2)


def get_EAR(image, outputs, P):
	landmark = outputs.multi_face_landmarks[0]

	P1 = landmark.landmark[P[0]]
	P2 = landmark.landmark[P[1]]
	P3 = landmark.landmark[P[2]]
	P4 = landmark.landmark[P[3]]
	P5 = landmark.landmark[P[4]]
	P6 = landmark.landmark[P[5]]

	a = euclidean_distance(image, P2, P6)
	b = euclidean_distance(image, P3, P5)
	c = euclidean_distance(image, P1, P4)

	return (a + b) / (2 * c)


if __name__ == '__main__':
	face_mesh = mp.solutions.face_mesh
	draw_utils = mp.solutions.drawing_utils
	mp_face_mesh2 = mp.solutions.face_mesh
	face_mesh2 = mp_face_mesh2.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
	mp_hand = mp.solutions.hands
	hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )
	

	mp_drawing = mp.solutions.drawing_utils

	drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
	COLOR_RED = (0, 0, 255)
	COLOR_BLUE = (255, 0, 0)
	COLOR_GREEN = (0, 255, 0)

	# RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
	# LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

	LEFT_EYE_IND = [362, 385, 387, 263, 373, 380]
	RIGHT_EYE_IND = [33, 160, 158, 133, 153, 144]

	FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

	face_model = face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

	capture = cv.VideoCapture(0)

	frame_count = 0
	frame_count_2 = 0
	min_frame = 6
	threshold = 0.2
	pause = False
	while True:
		print(pause)
		result, image = capture.read()
		

		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = face_mesh2.process(image)
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		#pause = True
		finger_count = 0
		
	

		img_h, img_w, img_c = image.shape
		re = hands.process(image)
		if re.multi_hand_landmarks:
			myHand = []
			count = 0
			for idx, hand in enumerate(re.multi_hand_landmarks):
				mp_drawing.draw_landmarks(image, hand, mp_hand.HAND_CONNECTIONS)
			for id, lm in enumerate(hand.landmark):
				h, w, _ = image.shape
				myHand.append([int(lm.x * w), int(lm.y * h)]) ## x= 0 , y =1
			if myHand[8][1] < myHand[5][1]:
				count = count + 1
			if myHand[4][0] < myHand[2][0]:
				count = count + 1
			if myHand[12][1] < myHand[9][1]:
				count = count + 1
			if myHand[16][1] < myHand[13][1]:
				count = count + 1
			if myHand[20][1] < myHand[17][1]:
				count = count + 1
			if count == 1:
				cv2.putText(image, "PAUSE", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
				pause = False
			if count == 5:
				pause = True
		if pause == True:
			face_3d = []
			face_2d = []

			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					for idx, lm in enumerate(face_landmarks.landmark):
						if idx in [33, 263, 1, 61, 291, 199]:
							if idx == 1:
								nose_2d = (lm.x * img_w, lm.y * img_h)
								nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
	
							x, y = int(lm.x * img_w), int(lm.y * img_h)
	
							# Get the 2D Coordinates
							face_2d.append([x, y])
	
							# Get the 3D Coordinates
							face_3d.append([x, y, lm.z])       
	
					# Convert it to the NumPy array
					face_2d = np.array(face_2d, dtype=np.float64)
	
					# Convert it to the NumPy array
					face_3d = np.array(face_3d, dtype=np.float64)
	
					# The camera matrix
					focal_length = 1 * img_w
	
					cam_matrix = np.array([ [focal_length, 0, img_h / 2],
											[0, focal_length, img_w / 2],
											[0, 0, 1]])
	
					# The distortion parameters
					dist_matrix = np.zeros((4, 1), dtype=np.float64)
	
					# Solve PnP
					success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
	
					# Get rotational matrix
					rmat, jac = cv2.Rodrigues(rot_vec)
	
					# Get angles
					angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
	
					# Get the y rotation degree
					x = angles[0] * 360
					y = angles[1] * 360
					z = angles[2] * 360
	
	
					if x < -10:
						frame_count_2 +=1
					else:
						frame_count_2 = 0
	
					nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
	
					p1 = (int(nose_2d[0]), int(nose_2d[1]))
					p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
	
					cv2.line(image, p1, p2, (255, 0, 0), 3)
					cv2.putText(
						image,
						f"x: {str(np.round(x, 2))}",
						(500, 50),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,
						(0, 0, 255),
						2,
					)
	
					if frame_count_2 > min_frame:
						message = 'Drowsiness detected'
						image = cv.putText(image, message, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 1, cv.LINE_AA)
						try:
							t1 = threading.Thread(target=playsounds)
							t1.start()
						except:
							print('error')
	
			if result and frame_count_2 <= min_frame:
				image.flags.writeable = False
				rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
				outputs = face_model.process(rgb_img)
				image.flags.writeable = True
	
				if outputs.multi_face_landmarks:
					draw_landmarks(image, outputs, FACE, COLOR_GREEN)
	
					draw_landmarks(image, outputs, LEFT_EYE_IND, COLOR_RED)
					ratio_left = get_EAR(image, outputs, LEFT_EYE_IND)
	
					draw_landmarks(image, outputs, RIGHT_EYE_IND, COLOR_RED)
					ratio_right = get_EAR(image, outputs, RIGHT_EYE_IND)
	
					ratio = (ratio_left + ratio_right) / 2.0
					image = cv.putText(
						image,
						f"EAR = {str(ratio)}",
						(100, 100),
						cv.FONT_HERSHEY_SIMPLEX,
						1,
						COLOR_RED,
						2,
						cv.LINE_AA,
					)
	
					if ratio < threshold:
						frame_count += 1
					else:
						frame_count = 0
	
					if frame_count > min_frame: 
						# Closing the eyes
						message = 'Drowsiness detected'
						image = cv.putText(image, message, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 1, cv.LINE_AA)
						try:
							t1 = threading.Thread(target=playsounds)
							t1.start()
						except Exception:
							print('error')
									# mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         landmark_drawing_spec=drawing_spec,
            #         connection_drawing_spec=drawing_spec)
		cv.imshow("FACE MESH", image)
		if cv.waitKey(1) & 255 == 27:
			break

	capture.release()
	cv.destroyAllWindows()
