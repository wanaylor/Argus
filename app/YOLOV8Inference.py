import pdb
import onnxruntime
import time
import cv2
import numpy as np
import nncf
import os
from PIL import Image
import openvino as ov
from matplotlib.pyplot import imshow
from openvino import inference_engine as IE
from scipy import special
import colorsys
import random

class YOLOV8Inference(object):

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)
        # Create a list of colors for each class where each color is a tuple of 3 integer values
        self.rng = np.random.default_rng(3)
        self.colors = self.rng.uniform(0, 255, size=(len(self.class_names), 3))


    def __call__(self, frame):
        #pdb.set_trace()
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(converted)
        self.image = converted
        return self.detect_objects(self.image)

    def initialize_model(self, path):

        # Onnx session
        #sess_options = onnxruntime.SessionOptions()
        #sess_options.enable_profiling = True
        self.session = onnxruntime.InferenceSession(path, #sess_options,
                                                    providers=onnxruntime.get_available_providers())

        # OpenVino inference request
        #core = ov.Core()
        #model = core.read_model(model='./models/yolov8f/yolov8f.xml', weights='./models/yolov8f/yolov8f.bin')
        #compiled_model = core.compile_model(model, "CPU")
        #self.infer_request = compiled_model.create_infer_request()

        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)



        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (640,640))
        #input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        # Onnx session
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        """
        # OpenVino Inference
        start = time.time()
        self.infer_request.set_input_tensor(ov.Tensor(input_tensor))
        end = time.time()
        print(f'Setting input tensor took {end-start}s')
        start = time.time()
        self.infer_request.infer()
        end = time.time()
        print(f'Inference took {end-start}s')
        #pdb.set_trace()
        outputs = [np.array(self.infer_request.get_output_tensor().data)]
        #outputs = [self.infer_request.get_output_tensor(i).data for i in range(1)]
        """
        #prof_file = self.session.end_profiling()
        #print(prof_file)

#        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, draw_scores=True, mask_alpha=0.4):

        return self.draw_detections_tool(self.image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):

        unique_class_ids = np.unique(class_ids)

        keep_boxes = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices,:]
            class_scores = scores[class_indices]

            class_keep_boxes = self.nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])

        return keep_boxes

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou


    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


    def draw_detections_tool(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = self.draw_masks(det_img, boxes, class_ids, mask_alpha)

        # Draw bounding boxes and labels of detections
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = self.colors[class_id]

            self.draw_box(det_img, box, color)

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            self.draw_text(det_img, caption, box, color, font_size, text_thickness)
        out_frame = cv2.cvtColor(np.array(det_img), cv2.COLOR_RGB2BGR)

        return out_frame


    def draw_box(self, image: np.ndarray, box: np.ndarray, color= (0, 0, 0, 255),
                 thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


    def draw_text(self, image: np.ndarray, text: str, box: np.ndarray, color= (0, 0, 0, 255),
                  font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(image, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    def draw_masks(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for box, class_id in zip(boxes, classes):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

if __name__ == '__main__':

    pdb.set_trace()
    model_path = "./models/yolov8n.onnx"
    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOV8Inference(model_path, conf_thres=0.3, iou_thres=0.5)

    img_url = "./images/image.jpg"
    img = cv2.imread(img_url)

    # Detect Objects
    yolov8_detector(img)

    # Draw detections
    combined_img = yolov8_detector.draw_detections()
    pdb.set_trace()
    print('all done')
