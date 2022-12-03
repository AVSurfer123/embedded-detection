import cv2
import argparse
import numpy as np
from capture import *
from model import *

parser = argparse.ArgumentParser(description="Main script.", formatter_class=lambda prog: argparse.HelpFormatter(prog))

parser.add_argument('input_type', type=str, default="video")
parser.add_argument('debug', type=bool, default=False)

args, leftovers = parser.parse_known_args()

if args.input_type == "video":
    frames = capture_video()
elif args.input_type == "webcam" or args.input_type == "camera":
    frames = capture_webcam()

print(frames.shape)
# run_model(frames, args.debug)
run_model2(frames)