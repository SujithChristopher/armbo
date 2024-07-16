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

import multiprocessing
from threading import Thread

import getopt
import argparse
import logging
import time
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()
h = 720
w = 1280
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True

        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# config.enable_stream(rs.stream.depth, w, h,rs.format.z16, 30)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)


pipeline.start(config)


class RecordData:
    def __init__(self, _pth=None, record_camera=True, fps_value=30):
        """webcam parameters for recording"""
        self.record_camera = record_camera
        self.start_recording = False
        self._pth = _pth
        self.kill_signal = False
        self.fps_val = fps_value
        self.display = True

    def capture_webcam(self):
        """capture webcam"""
        profile = pipeline.get_active_profile()

        # list available webcam
        frames = pipeline.wait_for_frames()

        if self.record_camera:
            _save_pth = os.path.join(self._pth, "webcam_color.msgpack")
            _save_file = open(_save_pth, "wb")
            _timestamp_file = open(
                os.path.join(self._pth, "webcam_timestamp.msgpack"), "wb"
            )
            print("im here")

        # prev_frame_time = 0   # for fps display
        # new_frame_time = 0

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            # if frame:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('webcam', gray_image)

            if self.record_camera and self.start_recording:
                _packed_file = mp.packb(gray_image, default=mpn.encode)
                _save_file.write(_packed_file)

            if self.display:
                cv2.imshow("webcam", gray_image)
                cv2.waitKey(1)

            if keyboard.is_pressed("q"):  # if key 'q' is pressed
                print("You Pressed a Key!, ending webcam")
                pipeline.stop()
                cv2.destroyAllWindows()
                # self.kill_thread()  # finishing the loop
                if self.record_camera:
                    _save_file.close()
                    _timestamp_file.close()
                # sys.exit()
                break

            if keyboard.is_pressed("s"):  # if key 's' is pressed
                print("You Pressed a Key!, started recording from webcam")
                self.start_recording = True

    def run(self, cart_sensors):
        """run the program"""
        # run the program
        self.capture_webcam()


if __name__ == "__main__":
    """get parameter from external program"""

    parser = argparse.ArgumentParser(
        prog="Single camera recorder",
        description="This basically records data from the camera and the sensors",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument("-s", "--sensors", help="record sensors", required=False)

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
        _pth = None  # this is default do not change, path gets updated by your input
        _folder_name = (
            "paper"  # this is the parent folder name where the data will be saved
        )

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
        _pth = os.path.join(
            os.path.dirname(__file__), "..", "recorded_data", _folder_name, _name
        )
        print(_pth)

        if "\n" in _pth:
            _pth = _pth.replace("\n", "")

        if not os.path.exists(_pth):
            os.makedirs(_pth)
    time.sleep(1)

    record_data = RecordData(_pth=_pth, record_camera=record_camera)
    record_data.run(cart_sensors=record_sensors)
