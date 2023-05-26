"""
this program records data from webcam and teensy controller 
"""
import cv2
import os
import sys
import datetime
import keyboard
import msgpack as mp
import msgpack_numpy as mpn
import fpstimer
import multiprocessing
from threading import Thread
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sensors import SerialPort
from support.pymf import get_MF_devices as get_camera_list
# import support.pymf
import getopt
import argparse
import logging
import time

class RecordData:
    def __init__(self, _pth = None, record_camera = True, fps_value = 30, isColor = False):

        self.device_list = get_camera_list()
        # print(self.device_list)
        self.cam_device = self.device_list.index("Lenovo FHD Webcam")

        """webcam parameters for recording"""
        self.yResRs = 640
        self.xResRs = 640

        self.xPos = 320 # fixed parameters
        self.yPos = 40
        self.record_camera = record_camera
        self.start_recording = False
        self._pth = _pth
        self.kill_signal = False
        self.fps_val = fps_value
        self.display = True

        self.isColor = isColor
    
    def capture_webcam(self):
        """capture webcam"""

        #list available webcam
        cap = cv2.VideoCapture(self.cam_device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        cap.set(cv2.CAP_PROP_FPS, self.fps_val)

        if self.record_camera:
            _save_pth = os.path.join(self._pth, "webcam_color.msgpack")
            _save_file = open(_save_pth, "wb")
            _timestamp_file = open(os.path.join(self._pth, "webcam_timestamp.msgpack"), "wb")

        # prev_frame_time = 0   # for fps display
        # new_frame_time = 0

        while True:
            ret, frame = cap.read() 
            if ret:
                if self.isColor:
                    gray_image = frame[self.yPos:self.yPos + self.yResRs, self.xPos:self.xPos + self.xResRs].copy()
                    # gray_image = frame.copy()
                else:
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_image = gray_image[self.yPos:self.yPos + self.yResRs, self.xPos:self.xPos + self.xResRs].copy()
                    # gray_image = gray_image.copy()

                if self.record_camera and self.start_recording:
                    _packed_file = mp.packb(gray_image, default=mpn.encode)
                    _save_file.write(_packed_file)
                    _time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    _packed_timestamp = mp.packb(_time_stamp)
                    _timestamp_file.write(_packed_timestamp)

                fpstimer.FPSTimer(self.fps_val)

                if self.display:
                    # # fps display
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # new_frame_time = time.time()
                    # fps = 1/(new_frame_time-prev_frame_time)
                    # prev_frame_time = new_frame_time
                    # fps = int(fps)
                    # fps = str(fps)
                    # cv2.putText(gray_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                    # fps display

                    cv2.imshow('webcam', gray_image)
                    cv2.waitKey(1)

                if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                    print('You Pressed a Key!, ending webcam')
                    cap.release()
                    cv2.destroyAllWindows()                
                    # self.kill_thread()  # finishing the loop
                    if self.record_camera:
                        _save_file.close()
                        _timestamp_file.close()
                    # sys.exit()
                    break
                
                if keyboard.is_pressed('s'):  # if key 's' is pressed
                    print('You Pressed a Key!, started recording from webcam')
                    self.start_recording = True


    def run(self, cart_sensors):
        """run the program"""
        # run the program

        if not cart_sensors and self.record_camera:

            webcam_capture_frame = multiprocessing.Process(target=self.capture_webcam)
            webcam_capture_frame.start()
            webcam_capture_frame.join()

            if self.kill_signal:
                print("killing the process")
            
        if cart_sensors and not self.record_camera:

            myport = SerialPort("COM5", 115200, csv_path=self._pth, csv_enable=True, single_file_protocol=True, dof=9)
            cart_sensors = Thread(target=myport.run_program)
            cart_sensors.start()

        if cart_sensors and self.record_camera:

            myport = SerialPort("COM5", 115200, csv_path=self._pth, csv_enable=True, single_file_protocol=True)
            cart_sensors = Thread(target=myport.run_program)
            webcam_capture_frame = multiprocessing.Process(target=self.capture_webcam)
            
            cart_sensors.start()
            webcam_capture_frame.start()
            
            cart_sensors.join()
            webcam_capture_frame.join()
    
            if self.kill_signal:
                print("killing the process")

if __name__ == "__main__":

    """get parameter from external program"""

    parser = argparse.ArgumentParser(
                    prog = 'Single camera recorder',
                    description = 'This basically records data from the camera and the sensors',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('-f', '--folder', help='folder name', required=False)
    parser.add_argument('-n', '--name', help='name of the file', required=False)
    parser.add_argument('-c', '--camera', help='record camera', required=False)
    parser.add_argument('-s', '--sensors', help='record sensors', required=False)

    args = parser.parse_args()

    # if your not passing any arguments then the default values will be used
    # and you may have to enter the folder name and the name of the recording
    if not any(vars(args).values()):
        print("No arguments passed, please enter manually")

        """Enter the respective parameters"""
        record_camera = True
        record_sensors = False

        if record_camera or record_sensors:
            _name = input("Enter the name of the recording: ")
        display = True
        _pth = None # this is default do not change, path gets updated by your input
        _folder_name = "cam_may_25_5_2023" # this is the parent folder name where the data will be saved

    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera
        record_sensors = args.sensors

        if record_camera == "True":
            record_camera = True
        else:
            record_camera = False
        if record_sensors == "True":
            record_sensors = True
        else:
            record_sensors = False

    if record_camera or record_sensors:
        _pth = os.path.join(os.path.dirname(__file__),"..", "recorded_data",_folder_name, _name)

        if '\n' in _pth:
            _pth = _pth.replace('\n', '')

        if not os.path.exists(_pth):
            os.makedirs(_pth)
    time.sleep(1)

    record_data = RecordData(_pth=_pth, record_camera=record_camera, isColor=True)
    record_data.run(cart_sensors=record_sensors)