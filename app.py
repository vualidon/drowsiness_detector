import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 as cv
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dis
import threading
from main import draw_landmarks, euclidean_distance, get_EAR
from playsound import playsound
import random

##
from streamlit_webrtc import VideoHTMLAttributes
from audio import AudioFrameHandler
##

##

audio_handler = AudioFrameHandler(sound_file_path="wakeup.mp3")

lock = threading.Lock()  # For thread-safe access
shared_state = {"play_alarm": False}
##

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
mp_face_mesh2 = mp.solutions.face_mesh
face_mesh2 = mp_face_mesh2.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
model_complexity=0,
min_detection_confidence=0.55,
min_tracking_confidence=0.55
)


mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)

LEFT_EYE_IND = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IND = [33, 160, 158, 133, 153, 144]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("DROWSINESS DETECTION")
st.write("")

threshold1 = st.slider("Threshold (For x value of head pose)",min_value=-10.0,max_value=-1.0,step=0.1,value=-3.0)
threshold2 = st.slider("EAR",min_value=0.15,max_value=0.25,step=0.001,value=0.16)
min_frame = st.slider("MIN FRAME",min_value=10,max_value=100,step=1,value=30)

def gen_frames(image,frame_count,frame_count_2,_continue):
    global min_frame
    global threshold1
    global threshold2

    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh2.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    #_continue = True
    finger_count = 0

    img_h, img_w, img_c = image.shape
    re = hands.process(image)
    if re.multi_hand_landmarks:
        myHand = []
        count = 0
        for hand in re.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand, mp_hand.HAND_CONNECTIONS)
        for lm in hand.landmark:
            h, w, _ = image.shape
            myHand.append([int(lm.x * w), int(lm.y * h)]) ## x= 0 , y =1
        if myHand[8][1] < myHand[5][1]:
            count += 1
        if myHand[4][0] < myHand[2][0]:
            count += 1
        if myHand[12][1] < myHand[9][1]:
            count += 1
        if myHand[16][1] < myHand[13][1]:
            count += 1
        if myHand[20][1] < myHand[17][1]:
            count += 1
        if count == 1 or count == 0:
            cv.putText(image, "PAUSE", (50, 450), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
            _continue = False
        elif count == 5:
            _continue = True
    if _continue == True:
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
                success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360


                if x < threshold1:
                    frame_count_2 +=1
                else:
                    frame_count_2 = 0

                nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]) , int(nose_3d_projection[0][0][1]))

                cv.line(image, p1, p2, (255, 0, 0), 3)
                cv.putText(
                    image,
                    f"x: {str(np.round(x, 2))}",
                    (10, 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if frame_count_2 > min_frame:
                    message = 'Drowsiness detected'
                    image = cv.putText(image, message, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 1, cv.LINE_AA)
                    # try:
                    #     t1 = threading.Thread(target=playsounds)
                    #     t1.start()
                    # except:
                    #     print('error')






        if frame_count_2 <= min_frame:
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
                    f"EAR = {str(np.round(ratio, 2))}",
                    (10, 100),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLOR_RED,
                    2,
                    cv.LINE_AA,
                )

                if ratio < threshold2:
                    frame_count += 1
                else:
                    frame_count = 0

                if frame_count > min_frame: 
                    # Closing the eyes
                    message = 'Drowsiness detected'
                    image = cv.putText(image, message, (10, 180), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 1, cv.LINE_AA)
                    # try:

    # Trả về response chứa frame đã xử lý
    return image,frame_count,frame_count_2,_continue




frame_count = 0
frame_count_2 = 0
_continue = True
def callback(frame):
    global frame_count
    global frame_count_2
    global _continue
    img = frame.to_ndarray(format='bgr24')
    img,frame_count,frame_count_2,_continue = gen_frames(img,frame_count,frame_count_2,_continue)
    with lock:
        shared_state["play_alarm"] = (
            frame_count > min_frame or frame_count_2 > min_frame
        )
    return av.VideoFrame.from_ndarray(img,format='bgr24')

##
def audio_frame_callback(frame: av.AudioFrame):
    with lock:  # access the current “play_alarm” state
        play_alarm = shared_state["play_alarm"]

    new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_alarm)
    return new_frame
##

# webrtc_streamer(
#     key="example",
#     video_frame_callback=callback,
#     audio_frame_callback=audio_frame_callback,
#     rtc_configuration={  # Add this line
#         "iceServers": [{"urls": ["turn:relay1.expressturn.com:3478"],"username":"ef4BWIICOST30PU5D8","credential": "oGlk0iIjiJyEqgEb"}]
#     }
# )

# webrtc_streamer(
#     key="example",
#     video_frame_callback=callback,
#     audio_frame_callback=audio_frame_callback,
#     rtc_configuration={  # Add this line
#         "iceServers": [{"urls": ["turn:relay4.expressturn.com:3478"],"username":"efEH1S1SLKBBPJGW82","credential": "XVm8IFOVfcTTejFP"}]
#     }
# )

webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    audio_frame_callback=audio_frame_callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["turn:relay3.expressturn.com:80"],"username":"efEH1S1SLKBBPJGW82", "credential":"XVm8IFOVfcTTejFP", "Secret Key":"6ou0f5d86sq0yiqwguc8"}]
    }
)
