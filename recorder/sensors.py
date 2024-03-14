"""This program is for recording IMU data, through HC05 bluetooth module"""

from turtle import st
import serial
import struct
import keyboard
import csv
from datetime import datetime
import sys
from sys import stdout
import getopt
import time

from deep_vision import rs_time



class SerialPort(object):
    # Contains functions that enable communication between the docking station and the IMU watches

    def __init__(self, serialport, serialrate=9600, csv_path="", csv_enable=False, single_file_protocol=False, dof=9, csv_name=None):
        # Initialise serial payload
        self.count = 0
        self.plSz = 0
        self.payload = bytearray()

        self.serialport = serialport
        self.ser_port = serial.Serial(serialport, serialrate)
        self.dof = dof

        self.csv_enabled = csv_enable
        if csv_enable:
            if csv_name is None:
                self.csv_file = open(csv_path+ "//imu01.csv", "w")
            else:
                self.csv_file = open(csv_path+ "//" + str(csv_name) + ".csv", "w")
                
            self.csv = csv.writer(self.csv_file)
            if self.dof == 6:
                self.csv.writerow(["sys_time","rust_time" ,"e_fr", "e_fl", "e_rr", "e_rl", "rtc", "mils", "sync", "ax", "ay", "az", "gx", "gy", "gz"])
            elif self.dof == 9:
                self.csv.writerow(["sys_time","rust_time" ,"e_fr", "e_fl", "e_rr", "e_rl", "rtc", "mils", "sync", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"])
        self.triggered = True
        self.connected = False
        

        stdout.write("Initializing imu program\n")

    def serial_write(self, payload):
        # Format:
        # | 255 | 255 | no. of bytes | payload | checksum |

        header = [255, 255]
        chksum = 254
        payload_size = len(payload) + 1
        chksum += payload_size

        self.ser_port.write(bytes([header[0]]))
        self.ser_port.write(bytes([header[1]]))
        self.ser_port.write(bytes([payload_size]))
        self.ser_port.write(bytes([payload]))
        self.ser_port.write(bytes([chksum % 256]))

    def serial_read(self):
        """returns bool for valid read, also returns the data read"""

        if (self.ser_port.read() == b'\xff') and (self.ser_port.read() == b'\xff'):
            self.connected = True
            chksum = 255 + 255
            self.plSz = self.ser_port.read()[0]
            chksum += self.plSz
            self.payload = self.ser_port.read(self.plSz - 1)
            # print(self.payload)

            chksum += sum(self.payload)
            chksum = bytes([chksum % 256])
            _chksum = self.ser_port.read()

            return _chksum == chksum

        return False

    def disconnect(self):
        stdout.write("disconnected\n")

    def run_program(self):
        while True:
            if self.serial_read():
                # val = struct.unpack("4l", self.payload[:16])    # encoder values
                # _rtc = struct.unpack("Q", self.payload[16:24])    # rtc values time delta
                # mils = struct.unpack("L", self.payload[24:28])
                _sync = struct.unpack("c", self.payload)[0].decode("utf-8")
                print(_sync)
                # _imu_data = struct.unpack("6f", self.payload[29:53])
                # if len(self.payload) > 53:
                #     _magnetometer = struct.unpack("3f", self.payload[53:65])

                # sys.stdout.write("\r" + _sync)
                # sys.stdout.flush()

                # _rtcval = datetime.fromtimestamp(_rtc[0]).strftime("%Y-%m-%d %I.%M.%S.%f %p")

                # # time_delta = struct.unpack("3H", self.payload[24:30])
                rs = rs_time()

                nw = None

                if not nw:
                    nw = datetime.now()     # datetime

                if self.csv_enabled:
                    if self.dof == 6:
                        self.csv.writerow([str(nw), rs, val[0], val[1], val[2], val[3], _rtcval, mils[0], _sync, _imu_data[0], _imu_data[1], _imu_data[2], _imu_data[3], _imu_data[4], _imu_data[5]])
                    elif self.dof == 9:
                        self.csv.writerow([str(nw), rs, val[0], val[1], val[2], val[3], _rtcval, mils[0], _sync, _imu_data[0], _imu_data[1], _imu_data[2], _imu_data[3], _imu_data[4], _imu_data[5], _magnetometer[0], _magnetometer[1], _magnetometer[2]])
                if keyboard.is_pressed("e"):
                    self.csv_file.close()
                    break
            if keyboard.is_pressed("q"):
                print("closing")
                break


if __name__ == '__main__':
    # opts, args = getopt.getopt(sys.argv[1:], "p:", ["path"])

    # print(opts[0])
    # _filepath = opts[0][1]
    _filepath = r"D:\CMC\DeepVision\recorded_data\validation\test"

    # myport = SerialPort("COM15", 115200, csv_path=_filepath, csv_enable=True)
    myport = SerialPort("COM3", 115200, csv_path=_filepath, csv_enable=False, dof=9)
    # myport = SerialPort("COM4", 115200, csv_path="random", csv_enable=False)
    # myport = SerialPort("COM4", 115200)
    myport.run_program()

