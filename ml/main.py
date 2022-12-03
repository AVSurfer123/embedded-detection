import cv2
import argparse
import numpy as np
from capture import capture_video, capture_webcam
from model import run_model

parser = argparse.ArgumentParser(description="Main script.", formatter_class=lambda prog: argparse.HelpFormatter(prog))

parser.add_argument('input_type', type=str, default="video")
args, leftovers = parser.parse_known_args()

if args.input_type == "video":
    frames = capture_video()
    print(frames.shape)
    run_model(frames)
elif args.input_type == "webcam":
    capture_webcam()
