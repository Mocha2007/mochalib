# https://stackoverflow.com/a/59016932/2579798
import os
os.add_dll_directory(os.getcwd())
# try https://www.csestack.org/calling-c-functions-from-python/ for newton-raphsonfrom ctypes import *
from ctypes import CDLL
mochamath = CDLL(f"mochamath.dll", winmode=0)
newton_raphson = mochamath.newton_raphson
