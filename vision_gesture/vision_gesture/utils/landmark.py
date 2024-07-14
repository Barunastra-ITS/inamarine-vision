#!usr/bin/env python3
import cv2
import numpy as np
import copy
import itertools

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if finger_gesture_text != "":
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv2.LINE_AA)

    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Define finger connections (each finger is a sequence of landmark points)
        fingers = [
            (2, 3, 4),    # Thumb
            (5, 6, 7, 8), # Index finger
            (9, 10, 11, 12), # Middle finger
            (13, 14, 15, 16), # Ring finger
            (17, 18, 19, 20)  # Little finger
        ]
        
        # Draw lines for each finger
        for finger in fingers:
            for i in range(len(finger) - 1):
                cv2.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i+1]]), (0, 0, 0), 6)
                cv2.line(image, tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i+1]]), (255, 255, 255), 2)
        
        # Define palm connections
        palm = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
        
        # Draw lines for palm
        for (start, end) in palm:
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)

    # Draw key points
    for index, landmark in enumerate(landmark_point):
        fingertips = [4, 8, 12, 16, 20]
        radius = 8 if index in fingertips else 5  # Larger radius for fingertips
        cv2.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv2.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image
