from YOLOV8Inference import YOLOV8Inference
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from functools import wraps
import pdb
import datetime
import os
import threading
import argparse
import datetime
import requests
import time
import cv2

class Runtime(Flask):
    def __init__(self, last_detection_dict):
        super().__init__('Argus')
        self.add_url_rule('/video_feed', view_func=self.video_feed)
        self.add_url_rule('/', view_func=self.index)

        self.last_detection_dict = last_detection_dict
        self.outputFrame = None
        self.lock = threading.Lock()
        self.app = Flask(__name__)
        self.proto = os.getenv('UPSTREAM_PROTO')
        self.address = os.getenv('UPSTREAM_IP')
        self.port = os.getenv('UPSTREAM_PORT')
        self.upstream_route = os.getenv('UPSTREAM_ROUTE')
        self.conf_thres = float(os.getenv('CONF_THRES'))
        self.detection_reset_seconds = int(os.getenv('DETECTION_RESET_SECONDS'))
        self.url = os.getenv('UPSTREAM_URL')
        self.vs = cv2.VideoCapture(self.url)
        self.model = YOLOV8Inference('./models/yolov8n.onnx', conf_thres=self.conf_thres)

        self.tracking_valid = False
        self.tracker = cv2.TrackerMIL()

        t = threading.Thread(target=self.poll_frames)
        t.daemon = True
        t.start()
        print(f"{datetime.datetime.now()} Sleeping to start polling")
        time.sleep(2.0)
        t2 = threading.Thread(target=self.detect_motion, args=(
            args["frame_count"],))
        t2.daemon = True
        t2.start()

    def index(self):
        # return the rendered template
        return render_template("index.html")

    def notify(self, cls_ids, frame):
        for clss in cls_ids:
            obj = self.model.class_names[clss]
            if obj in self.last_detection_dict.keys():
                print(f"{datetime.datetime.now()} Got Detection! {obj}")
                if (time.time() - self.last_detection_dict[obj] > self.detection_reset_seconds): 
                    self.last_detection_dict[obj] = time.time()
                    start = time.time()
                    cv2.imwrite('detection.jpg', frame)
                    print(f"{datetime.datetime.now()} Writing detection.jpeg to disk took {time.time() - start}s")
                    print(f"{datetime.datetime.now()} Source inside notify is {self.vs.get(0)}")
                    r = requests.post("https://api.pushover.net/1/messages.json", data = {
                      "token": os.getenv('APP_TOKEN'),
                      "user": os.getenv('USER_TOKEN'),
                      "message": f"{obj} detected on {self.upstream_route}",
                    },
                    files = {
                      "attachment": ("image.jpg", open("detection.jpg", "rb"), "image/jpeg")
                    })

    def poll_frames(self):
        while True:
            start = time.time()
            ret, frame = self.vs.read()
            if ret == True:
                if frame is None:
                    print(f"{datetime.datetime.now()} Got None from frame, re-establishing video source")
                    self.vs.release()
                    self.vs = cv2.VideoCapture(f'{self.proto}://{self.address}:{self.port}/{self.upstream_route}')
                    continue
                try:
                    with self.lock:
                        self.outputFrame = frame.copy()
                    frame = None
                except Exception as e:
                    print(f"{datetime.datetime.now()} Error processing frame: {e}")
                    time.sleep(10)
                    continue
            else:
                print(f"{datetime.datetime.now()} Frame read didn't return True, restarting stream")
                self.vs.release()
                self.vs = cv2.VideoCapture(f'{self.proto}://{self.address}:{self.port}/{self.upstream_route}')

    def detect_motion(self, frameCount):
        total = 0

        while True:
            if self.outputFrame is None:
                time.sleep(10)
                continue
            with self.lock:
                frame = self.outputFrame.copy()
            start = time.time()
            boxes, scores, cls_ids = self.model(frame)
            frame = self.model.draw_detections()
            print(f"cls_ids is {cls_ids}")
            self.notify(cls_ids, frame)
            end = time.time()
            print(f'{datetime.datetime.now()} Detection took {end-start} s')
            self.outputFrame = frame
            frame = None
            time.sleep(1)


    def generate(self):
        while True:
            with self.lock:
                if self.outputFrame is None:
                    continue
                self.outputFrame = cv2.resize(self.outputFrame, (1600,900))
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame)
                if not flag:
                    continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

    def video_feed(self):
        return Response(self.generate(),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    ap.add_argument("-d", "--detections", nargs="*",
                    help="points of interest for detection i.e. person car")
    args = vars(ap.parse_args())
    detection_reset_seconds = int(os.getenv('DETECTION_RESET_SECONDS'))
    last_detection_dict = {x: time.time() - detection_reset_seconds for x in args['detections']}
    rt = Runtime(last_detection_dict=last_detection_dict)
    rt.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
