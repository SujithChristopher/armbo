"""This program is for recording IMU data, through HC05 bluetooth module"""

import serial
import struct
import keyboard
import csv
import sys
from sys import stdout
from deep_vision import rs_time


class SerialPort(object):
    # Contains functions that enable communication between the docking station and the IMU watches

    def __init__(
        self,
        serialport,
        serialrate=9600,
        csv_path="",
        csv_enable=False,
        csv_name=None,
    ):
        # Initialise serial payload
        self.count = 0
        self.plSz = 0
        self.payload = bytearray()

        self.serialport = serialport
        self.ser_port = serial.Serial(serialport, serialrate, timeout=0.5)

        self.csv_enabled = csv_enable
        if csv_enable:
            if csv_name is None:
                self.csv_file = open(csv_path + "//sync01.csv", "w")
            else:
                self.csv_file = open(csv_path + "//" + str(csv_name) + ".csv", "w")

            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["rust_time", "sync"])
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
        # print(self.ser_port.read())

        if (self.ser_port.read() == b"\xff") and (self.ser_port.read() == b"\xff"):
            self.connected = True
            chksum = 255 + 255
            self.plSz = self.ser_port.read()[0]
            chksum += self.plSz
            self.payload = self.ser_port.read(self.plSz - 1)
            chksum += sum(self.payload)
            chksum = bytes([chksum % 256])
            _chksum = self.ser_port.read()

            return _chksum == chksum

        return False

    def disconnect(self):
        stdout.write("disconnected\n")

    def run_program(self):
        while self.ser_port.is_open:
            if self.serial_read():
                _sync = struct.unpack("c", self.payload)[0]
                sys.stdout.write("\r" + _sync.decode("utf-8"))
                sys.stdout.flush()

                rs = rs_time()

                if self.csv_enabled:
                    self.csv.writerow(
                        [
                            rs,
                            _sync.decode("utf-8"),
                        ]
                    )
                if keyboard.is_pressed("e"):
                    self.csv_file.close()
                    break
            if keyboard.is_pressed("q"):
                print("closing")
                break
            if not self.ser_port.is_open:
                print("port closed")
                break

        print("program ended")


if __name__ == "__main__":
    _filepath = r"C:\Users\CMC\Documents\openposelibs\pose\skateboard_gui\recording_programs\test_data\test"
    myport = SerialPort("COM3", 115200, csv_path=_filepath, csv_enable=False)
    myport.run_program()
