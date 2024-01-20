from YOLOV8Inference import YOLOV8Inference
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import datetime
import os
import threading
import argparse
import datetime
import requests
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
proto = os.getenv('UPSTREAM_PROTO')
address = os.getenv('UPSTREAM_IP')
port = os.getenv('UPSTREAM_PORT')
route = os.getenv('UPSTREAM_ROUTE')
conf_thres = float(os.getenv('CONF_THRES'))
detection_reset_seconds = int(os.getenv('DETECTION_RESET_SECONDS'))
url = os.getenv('UPSTREAM_URL')
vs = cv2.VideoCapture(url)
model = YOLOV8Inference('./models/yolov8n.onnx', conf_thres=conf_thres)
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def notify(cls_ids, frame):
    global vs
    for clss in cls_ids:
        obj = model.class_names[clss]
        if obj in last_detection_dict.keys():
            print(f"{datetime.datetime.now()} Got Detection! {obj}")
            if (time.time() - last_detection_dict[obj] > detection_reset_seconds): 
                last_detection_dict[obj] = time.time()
                start = time.time()
                cv2.imwrite('detection.jpg', frame)
                print(f"{datetime.datetime.now()} Writing detection.jpeg to disk took {time.time() - start}s")
                print(f"{datetime.datetime.now()} Source inside notify is {vs.get(0)}")
                r = requests.post("https://api.pushover.net/1/messages.json", data = {
                  "token": os.getenv('APP_TOKEN'),
                  "user": os.getenv('USER_TOKEN'),
                  "message": f"{obj} detected on {route}",
                },
                files = {
                  "attachment": ("image.jpg", open("detection.jpg", "rb"), "image/jpeg")
                })

def poll_frames():
    global vs, outputFrame, lock
    while True:
        start = time.time()
        ret, frame = vs.read()
        if ret == True:
            if frame is None:
                print(f"{datetime.datetime.now()} Got None from frame, re-establishing video source")
                vs.release()
                vs = cv2.VideoCapture(f'{proto}://{address}:{port}/{route}')
                continue
            try:
                with lock:
                    outputFrame = frame.copy()
                print(f"Capture position is {vs.get(0)}")
                frame = None
            except Exception as e:
                print(f"{datetime.datetime.now()} Error processing frame: {e}")
                time.sleep(10)
                continue
        else:
            print(f"{datetime.datetime.now()} Frame read didn't return True, restarting stream")
            vs.release()
            vs = cv2.VideoCapture(f'{proto}://{address}:{port}/{route}')

def detect_motion(frameCount):
    global outputFrame, lock
    total = 0

    while True:
        if outputFrame is None:
            time.sleep(10)
            continue
        with lock:
            frame = outputFrame.copy()
        start = time.time()
        boxes, scores, cls_ids = model(frame)
        frame = model.draw_detections()
        print(f"cls_ids is {cls_ids}")
        notify(cls_ids, frame)
        end = time.time()
        print(f'{datetime.datetime.now()} Detection took {end-start} s')
        outputFrame = frame
        frame = None
        time.sleep(1)


def generate():
    global outputFrame, lock, model
    while True:
        with lock:
            if outputFrame is None:
                continue
            outputFrame = cv2.resize(outputFrame, (1600,900))
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
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
    last_detection_dict = {x: time.time() - detection_reset_seconds for x in args['detections']}
    t = threading.Thread(target=poll_frames)
    t.daemon = True
    t.start()
    print(f"{datetime.datetime.now()} Sleeping for a minute to start polling")
    t2 = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t2.daemon = True
    t2.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
