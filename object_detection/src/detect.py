#!/usr/bin/env python3

import rospy
import torch
import loguru
import numpy as np
import sys
import pickle

from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (Profile, check_file, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from PIL import Image as PILImage

from rospy import Publisher
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nala_msgs.msg import BoundingBox, BoundingBoxArray

class MainVision:
    def __init__(self):
        loguru.logger.remove(0)
        loguru.logger.add(sys.stderr, format = "<red>{level}</red> | <green>{message}</green>", colorize=True)

        self.cv_bridge = CvBridge()
        self.bbox_array = BoundingBoxArray()

        self.rate = rospy.Rate(30)
        self.img_pub = Publisher("/vision/image_raw", Image, queue_size=10)
        self.obj_pub = Publisher("/vision/object_raw", BoundingBoxArray, queue_size=10)
        self.clock = rospy.Time()

        self.weights = '/home/barun/inamarine/src/object_detection/src/trained/yolov5n_adam/yolov5n_adam_openvino_model'
        self.data = '/home/barun/inamarine/src/object_detection/src/trained/yolov5n_adam/yolov5n_adam_openvino_model/yolov5n_adam.yaml'

        self.source = '/home/barun/inamarine/src/object_detection/src/test/ball.mp4'
        self.imgsz = (640, 640)
        self.conf_thres = 0.7
        self.iou_thres = 0.5
        self.max_det = 1000
        self.device = 'cpu'
        self.view_img = True
        self.nosave = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.visualize = False
        self.line_thickness = 1
        self.hide_labels = False
        self.hide_conf = False
        self.half = False
        self.dnn = False
        self.vid_stride = 1
        self.svm_load = False
        self.svm_size = (24, 24)

    def prepareDetection(self):
        """
        Prepare detection
        Returns:
            source (str): source path
            save_img (bool): save image flag
            is_file (bool): is file flag
            is_url (bool): is url flag
            webcam (bool): webcam flag
            device (str): device
            model (DetectMultiBackend): model
            stride (int): stride
            names (list): names
            pt (bool): pt flag
            imgsz (int): image size
        """
        source = str(self.source)
        save_img = not self.nosave and not source.endswith('.txt')
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)
        return source, save_img, is_file, is_url, webcam, device, model, stride, names, pt, imgsz
    
    def streamSelect(self, source, webcam, imgsz, stride, pt):
        """
        Select stream type
        Args:
            source (str): source path
            webcam (bool): webcam flag
            imgsz (int): image size
            stride (int): stride
            pt (bool): pt flag
        Returns:
            dataset (LoadStreams or LoadImages): dataset            
        """
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
        return dataset
    
    def showFps(self, img, fps):
        """
        Show FPS on top left corner of the image
        Args:
            img (np.ndarray): image
            fps (float): fps value
        Returns:
            np.ndarray: image with fps
        """
        font = cv2.FONT_HERSHEY_PLAIN
        line = cv2.LINE_AA
        fps_text = 'FPS: {:.2f}'.format(fps)
        cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
        return img
    
    def getBuoyColor(self, img, x_min, y_min, x_max, y_max):
        """
        Get buoy color using SVM
        Args:
            img (np.ndarray): image
            x_min (float): x min value
            y_min (float): y min value
            x_max (float): x max value
            y_max (float): y max value
        Returns:
            str: buoy color
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil_rgb = PILImage.fromarray(img_rgb[int(y_min):int(y_max), int(x_min):int(x_max)])
        img_pil_rgb = img_pil_rgb.resize(self.svm_size)
        img_pil_rgb = img_pil_rgb.convert('RGB')
        img_pil_rgb = np.array(img_pil_rgb)

        flattened_img = img_pil_rgb.flatten()
        prediction_color = self.color_model.predict([flattened_img])

        if prediction_color[0] == 'blue':
            return 'blue_'
        elif prediction_color[0] == 'green':
            return 'green_'
        elif prediction_color[0] == 'orange':
            return 'orange_'
        elif prediction_color[0] == 'purple':
            return 'purple_'
        elif prediction_color[0] == 'yellow':
            return 'yellow_'
        
    def run(self):
        source, save_img, _, _, webcam, device, model, stride, names, pt, imgsz = self.prepareDetection()
        dataset = self.streamSelect(source, webcam, imgsz, stride, pt)
        seen, dt = 0, (Profile(), Profile(), Profile()) # delta time (data, nms, tot)
        # start_time = self.clock.now()
        for path, im, im0s, _ , s in dataset:
            with dt[0]: # data time
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]
            with dt[1]: # nms time
                pred = model(im, augment=self.augment, visualize=self.visualize)
            with dt[2]: # tot time
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det): # detections not empty
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # to original shape
                    for c in det[:, 5].unique(): # draw each class
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    self.bbox_array.bounding_boxes.clear()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = names[c] if self.hide_conf else f'{names[c]}'
                        x_min, y_min, x_max, y_max = xyxy
                        h = float(y_max - y_min)
                        l = float(x_max - x_min)
                        if device == 'cpu':
                            center_x = float(np.mean([x_min, x_max]))
                            center_y = float(np.mean([y_min, y_max]))
                        else:
                            center_x = float(torch.mean(torch.tensor([x_min, x_max])))
                            center_y = float(torch.mean(torch.tensor([y_min, y_max])))
                        
                        if label == 'ball':
                            if not self.svm_load:
                                filename = '/home/barun/inamarine/src/object_detection/src/trained/color_model/ball_model.pkl'
                                with open(filename, 'rb') as f:
                                    self.color_model = pickle.load(f)
                                self.svm_load = True
                            buoy_color = self.getBuoyColor(im0, x_min, y_min, x_max, y_max)
                            # print(buoy_color)
                            label = buoy_color + label

                        self.bbox = BoundingBox()
                        self.bbox.class_name = label
                        self.bbox.bounding_box.center.x = center_x
                        self.bbox.bounding_box.center.y = center_y
                        self.bbox.bounding_box.size_x = l
                        self.bbox.bounding_box.size_y = h
                        self.bbox_array.header.stamp = self.clock.now()
                        self.bbox_array.header.frame_id = 'camera'
                        self.bbox_array.bounding_boxes.append(self.bbox)

                        if save_img or self.view_img:
                            c = int(cls)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    self.obj_pub.publish(self.bbox_array)
                im0 = annotator.result()
                inference_time = dt[1].dt * 1E3
                fps = 1E3 / inference_time
                im0 = self.showFps(im0, fps)
                try:
                    self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(im0, 'bgr8'))
                except CvBridgeError as e:
                    print(e)
                if self.view_img:
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):
                        loguru.logger.warning('Exit via keyboard interrupt (^C) or "q"')
                        raise StopIteration
                if rospy.is_shutdown():
                    loguru.logger.warning('Exit via keyboard interrupt (^C) or "q"')
                    raise StopIteration
            inference_time = dt[1].dt * 1E3
            loguru.logger.info(f"{s}{'' if len(det) else '(no detections), '}{inference_time:.1f}ms, FPS: {1 / inference_time * 1E3:.1f}")
        t = tuple(x.t / seen * 1E3 for x in dt)
        loguru.logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

if __name__ == '__main__':
    rospy.init_node('object_detection_node')
    main_vision = MainVision()
    main_vision.run()
    rospy.spin()