# Inamarine Vision Gesture

## Introduction

This project is a part of the Inamarine 2024 project. The Inamarine 2024 is an exhibition event that will be held in Jakarta, Indonesia. The Inamarine Vision Gesture project is a computer vision project that uses the OpenCV and MediaPipe libraries to detect and recognize hand gestures. The project is designed to be used in the Inamarine 2024 event to control Nala Proteus arm robot using hand gestures.

## Installation

To install the Inamarine Vision Gesture project, you need to follow the steps below:

1. Create ROS2 workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone the repository

```bash
git clone https://github.com/Barunastra-ITS/inamarine-vision
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Build the project

```bash
cd ~/ros2_ws
colcon build
```

5. Source the project

```bash
source ~/ros2_ws/install/setup.bash
```

6. Move the weights file and the label file to the build directory

```bash
cp ~/ros2_ws/src/vision_gesture/vision_gesture/model/keypoint_classifier_8_class.tflite ~/ros2_ws/build/lib/vision_gesture/model/
cp ~/ros2_ws/src/vision_gesture/vision_gesture/model/keypoint_classifier_label.csv ~/ros2_ws/build/lib/vision_gesture/model/
```

7. Change the label directory in the `vision_gesture.py` file

```python
with open('{your_home_directory}/ros2_ws/build/vision_gesture/build/lib/vision_gesture/model/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_labels = [row[0] for row in csv.reader(f)]
```

8. And the weights file directory at `model/keypoint_classifier.py`

```python
def __init__(
        self,
        model_path='{your_home_directory}/ros2_ws/build/vision_gesture/build/lib/vision_gesture/model/keypoint_classifier_8_class.tflite',
        num_threads=1,
    ):
```

## Usage

To use the Inamarine Vision Gesture project, just run the `vision_gesture.py` file.

```bash
cd ~/ros2_ws
ros2 run vision_gesture vision_gesture
```

## Topics

1. `/gesture_recognition` (std_msgs/String)

The `/gesture_recognition` topic publishes the recognized gesture.

## Gesture List

The Inamarine Vision Gesture project can recognize 8 hand gestures. The gestures are:

1. fist (✊🏻)
2. god (☝🏻)
3. handsome (👉🏻 but with middle finger up)
4. ok (👌🏻)
5. palm (✋🏻)
6. peace (✌🏻)
7. pistol (👉🏻)
8. thumbs_up (👍🏻)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.