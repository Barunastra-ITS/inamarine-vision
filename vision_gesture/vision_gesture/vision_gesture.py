#!usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import csv
import mediapipe as mp
import copy
import rclpy
import time

from vision_gesture.model.keypoint_classifier import KeyPointClassifier
from vision_gesture.utils.landmark import calc_bounding_rect, calc_landmark_list, pre_process_landmark, draw_bounding_rect, draw_landmarks, draw_info_text
from rclpy.node import Node
from std_msgs.msg import String

class VisionGesture(Node):
    def __init__(self):
        super().__init__('vision_gesture')

        self.device = 0
        self.width = 640
        self.height = 480
        self.conf_th = 0.7
        self.max_num_hands = 1
        self.use_brect = True

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.conf_th,
        )

        self.keypoint_classifier = KeyPointClassifier()

        with open('/home/ikhw/inamarine_vision_gesture/build/vision_gesture/build/lib/vision_gesture/model/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_labels = [row[0] for row in csv.reader(f)]

        self.publisher_ = self.create_publisher(String, 'gesture_recognition', 10)

    def run(self):
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print('Error: failed to capture image')
                break
            
            frame = cv2.flip(frame, 1)
            debug_frame = copy.deepcopy(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame.flags.writeable = False
            results = self.hands.process(frame)
            frame.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_frame, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                    pre_process_landmark_list = pre_process_landmark(landmark_list)

                    inference_start = time.time()
                    hand_sign_id = self.keypoint_classifier(pre_process_landmark_list)
                    gesture_name = self.keypoint_labels[hand_sign_id]
                    inference_end = time.time() - inference_start
                    

                    debug_frame = draw_bounding_rect(self.use_brect, debug_frame, brect)
                    debug_frame = draw_landmarks(debug_frame, landmark_list)
                    debug_frame = draw_info_text(
                        image=debug_frame,
                        brect=brect,
                        handedness=handedness,
                        hand_sign_text=gesture_name,
                        finger_gesture_text=""
                    )
                    debug_frame = cv2.putText(debug_frame, f'FPS: {0.001 / inference_end:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    msg = String()
                    msg.data = gesture_name
                    self.publisher_.publish(msg)

            else:
                debug_frame = cv2.putText(debug_frame, 'No hands detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hand Tracking', debug_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def main(self):
        self.run()

def main(args=None):
    rclpy.init(args=args)
    vision_gesture = VisionGesture()
    vision_gesture.main()
    rclpy.shutdown()
