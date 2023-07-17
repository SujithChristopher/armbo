import numpy as np
import cv2
import os
import msgpack as mp
import msgpack_numpy as mpn
from tqdm import tqdm
from cv2 import aruco
# from ultralytics import YOLO

_pth = os.getcwd()
# _pth = os.path.dirname(_pth)
_parent_folder = "cam_june_22_2023"
_calib_folder_name = "calibration_00"

_folder_name = "sk40_30_4_rotation_mocap_7"
# _folder_name = "board"

_base_pth = os.path.join(_pth,"recorded_data",_parent_folder)

_webcam_calib_folder = os.path.join(_pth,"recorded_data",_parent_folder,_calib_folder_name)
_webcam_calib_folder = os.path.join(_webcam_calib_folder)
_webcam_calib_pth = os.path.join( _webcam_calib_folder, "webcam_calibration.msgpack")

with open(_webcam_calib_pth, "rb") as f:
    webcam_calib = mp.Unpacker(f, object_hook=mpn.decode)
    _temp = next(webcam_calib)
    _webcam_cam_mat = _temp[0]
    _webcam_dist = _temp[1]

ar_lframe_pth = os.path.join(_webcam_calib_folder, "webcam_rotmat_2.msgpack")
with open(ar_lframe_pth, "rb") as f:
    ar_lframe = mp.Unpacker(f, object_hook=mpn.decode)
    _ar_lframe_rot = next(ar_lframe)
    _ar_lframe_org = next(ar_lframe)

# Aruco parameters
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
markerLength = 0.04
markerSeparation = 0.01

board = aruco.GridBoard_create(
        markersX=1,
        markersY=1,
        markerLength=markerLength,
        markerSeparation=markerSeparation,
        dictionary=ARUCO_DICT)

# Select camera
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
cap.set(cv2.CAP_PROP_FPS, 30)

rA_m1 = np.array([0.02, 0., -0.115]).reshape(3, 1)
translation_correction = np.array([0.0, 0.0, 0.0])

while True:
    # Capture frame-by-frame
    ret, _frame = cap.read()
    # Our operations on the frame come here
    if ret:    
    
        gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = gray,
            board = board,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            cameraMatrix = _webcam_cam_mat,
            distCoeffs = _webcam_dist)

        rotation_vectors, translation_vectors, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, _webcam_cam_mat, _webcam_dist)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 185)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2

        try:
            
            rot_i = cv2.Rodrigues(rotation_vectors[0][0])[0]
                        
            _frame = cv2.putText(_frame, f"p1 C x :{round(translation_vectors[0][0][0], 3)}", (5, 20), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"p1 C y :{round(translation_vectors[0][0][1], 3)}", (5, 45), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"p1 C z :{round(translation_vectors[0][0][2], 3)}", (5, 70), font, fontScale, color, thickness, cv2.LINE_AA, False)
            
            tv = translation_vectors[0][0].reshape(3,1) # reshape to 3x1

            # aruco in camera frame
            pb_C = (_ar_lframe_rot.T @ (translation_vectors[0][0].reshape(3,1) - _ar_lframe_org)).T[0]
            
            _frame = cv2.putText(_frame, f"rm1 A :{rot_i[0]}", (5, 100), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"rm1 A :{rot_i[1]}", (5, 130), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"rm1 A :{rot_i[2]}", (5, 160), font, fontScale, color, thickness, cv2.LINE_AA, False)
            # _frame = cv2.putText(_frame, f"Pb C x :{round(pb_C[0], 3)}", (5, 190), font, fontScale, color, thickness, cv2.LINE_AA, False)
            # _frame = cv2.putText(_frame, f"Pb C y :{round(pb_C[1], 3)}", (5, 220), font, fontScale, color, thickness, cv2.LINE_AA, False)
            # _frame = cv2.putText(_frame, f"Pb C z :{round(pb_C[2], 3)}", (5, 250), font, fontScale, color, thickness, cv2.LINE_AA, False)
            
            p_A_c = (rot_i @ rA_m1 + tv) # 3x1 point in Aruco frame
            
            for_display = p_A_c.T[0] # for display only
            
            _frame = cv2.putText(_frame, f"Pa C x :{round(for_display[0], 3)}", (5, 280), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"Pa C y :{round(for_display[1], 3)}", (5, 310), font, fontScale, color, thickness, cv2.LINE_AA, False)
            _frame = cv2.putText(_frame, f"Pa C z :{round(for_display[2], 3)}", (5, 340), font, fontScale, color, thickness, cv2.LINE_AA, False)
            
            p_A_b = _ar_lframe_rot.T @ (p_A_c - _ar_lframe_org ).T[0]
            
            # _frame = cv2.putText(_frame, f"Pa B :{round(p_A_b[0], 3)}", (5, 380), font, fontScale, color, thickness, cv2.LINE_AA, False)
            # _frame = cv2.putText(_frame, f"Pa B :{round(p_A_b[1], 3)}", (5, 410), font, fontScale, color, thickness, cv2.LINE_AA, False)
            # _frame = cv2.putText(_frame, f"Pa B :{round(p_A_b[2], 3)}", (5, 430), font, fontScale, color, thickness, cv2.LINE_AA, False)
            cv2.drawFrameAxes(_frame, _webcam_cam_mat, _webcam_dist, rotation_vectors, translation_vectors, 0.04)
        except:
            pass
        # this is for black and white image
        
        cv2.imshow('frame',_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
