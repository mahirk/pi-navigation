# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import qwiic
from numpy import ones, vstack
from numpy.linalg import lstsq
import math
from scipy.optimize import curve_fit
import qwiic_button

my_button = qwiic_button.QwiicButton()
if my_button.begin() == False:
    print(
        "\nThe Qwiic Button isn't connected to the system. Please check your connection"
    )


cap = cv2.VideoCapture(0)

print("VL53L1X Qwiic Test\n")
ToF = qwiic.QwiicVL53L1X()
if ToF.sensor_init() == None:  # Begin returns 0 on a good init
    print("Sensor online!\n")

# Check slope for navigation
def checkSlope(x1, y1, x2, y2):
    if not (x1 == x2):
        a = False if ((y2 - y1) / (x2 - x1)) < 0 else True
        return a


# Finding the intersection of 2 lines
def intersection(line, line1):
    x = (line1[1] - line[1]) / (line[0] - line1[0])
    if (
        (x <= 800 and x >= 0)
        and (((line[0] * x) + line[1]) >= 0)
        and ((line[0] * x) + line[1] <= 600)
    ):
        return [x, (line[0] * x) + line[1]]


# Getting the slope of a line
def getSlope(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x2)
    return m


# Performing quadratic regression to outline lanes
def polyReg(xcors, ycors):
    def func(x, a, b, c):
        return (a * (x ** 2)) + (b * x) + c

    time = np.array(xcors)
    avg = np.array(ycors)
    initialGuess = [5, 5, -0.01]
    guessedFactors = [func(x, *initialGuess) for x in time]
    popt, pcov = curve_fit(func, time, avg, initialGuess)
    cont = np.linspace(min(time), max(time), 50)
    fittedData = [func(x, *popt) for x in cont]

    xcors = []
    ycors = []
    for count, i in enumerate(cont):
        xcors.append(i)
        ycors.append(fittedData[count])

    return popt, xcors, ycors


# Performing linear regression to outline the sides of lanes
def linearRegression(X, Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)

    x = np.linspace(np.min(X) - 5, np.max(X) + 5, 1000)
    y = b0 + b1 * x

    return b1, b0


# Averaging all the points for averaging lanes
def average(diction):
    xcors1 = 0
    ycors1 = 0
    xcors2 = 0
    ycors2 = 0
    count = 0
    for data in diction:
        xcors1 = xcors1 + data[2][0]
        ycors1 = ycors1 + data[2][1]
        xcors2 = xcors2 + data[2][2]
        ycors2 = ycors2 + data[2][3]
        count = count + 1
    xcors1 = xcors1 / count
    ycors1 = ycors1 / count
    xcors2 = xcors2 / count
    ycors2 = ycors2 / count

    return (int(xcors1), int(ycors1), int(xcors2), int(ycors2))


# Function to average houghlines for lanes
def averageLanes(lines):
    try:
        ycor = []

        for i in lines:
            for x in i:
                ycor.append(x[1])
                ycor.append(x[3])

        minY = min(ycor)
        maxY = 600
        linesDict = {}
        finalLines = {}
        lineCount = {}
        for count, i in enumerate(lines):
            for x in i:
                xcors = (x[0], x[2])
                ycors = (x[1], x[3])

                A = vstack([xcors, ones(len(xcors))]).T
                m, b = lstsq(A, ycors)[0]

                x1 = (minY - b) / m
                x2 = (maxY - b) / m

                linesDict[count] = [m, b, [int(x1), minY, int(x2), maxY]]

        status = False
        for i in linesDict:
            finalLinesCopy = finalLines.copy()

            m = linesDict[i][0]
            b = linesDict[i][1]

            line = linesDict[i][2]

            if len(finalLines) == 0:
                finalLines[m] = [[m, b, line]]
            else:
                status = False

                for x in finalLinesCopy:
                    if not status:
                        if abs(x * 1.2) > abs(m) > abs(x * 0.8):
                            if (
                                abs(finalLinesCopy[x][0][1] * 1.2)
                                > abs(b)
                                > abs(finalLinesCopy[x][0][1] * 0.8)
                            ):
                                finalLines[x].append([m, b, line])
                                status = True
                                break

                        else:
                            finalLines[m] = [[m, b, line]]

        for i in finalLines:
            lineCount[i] = len(finalLines[i])

        extremes = sorted(lineCount.items(), key=lambda item: item[1])[::-1][:2]
        lane1 = extremes[0][0]
        lane2 = extremes[1][0]

        l1x1, l1y1, l1x2, l1y2 = average(finalLines[lane1])
        l2x1, l2y1, l2x2, l2y2 = average(finalLines[lane2])

        allxcors = [[l1x1, l1x2], [l2x1, l2x2]]
        allycors = [[l1y1, l1y2], [l2y1, l2y2]]

        return allxcors, allycors

    except Exception as e:
        pass


# Cutting out the reigon of interest
def roi(img, vert):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vert, 255)
    return cv2.bitwise_and(img, mask)


# Filtering points
def getPoints(lines):
    xcors = []
    ycors = []
    for i in lines:
        xcors.append(i[0][0])
        ycors.append(i[0][1])
        xcors.append(i[0][2])
        ycors.append(i[0][3])
    return xcors, ycors


# Performing edge detection
def edgeDetect(img):
    edges = cv2.Canny(img, 250, 300)
    return cv2.GaussianBlur(edges, (3, 3), 0)


def write_and_print(text):
    print(text)
    with open("write.txt", "w") as f:
        f.write(text)


# Main function
def run(screen):
    vert = np.array([[100, 550], [375, 350], [450, 350], [800, 550]], np.int32)

    fin = edgeDetect(screen)
    fin = roi(fin, [vert])
    distance = ToF.get_distance()
    distance_is_too_close = False
    distanceFeet = distance / (25.4 * 12.0)
    cv2.putText(
        screen,
        f'Distance: {"{:.2f}".format(distanceFeet)}',
        (40, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 0),
        3,
    )

    if distanceFeet < 2:
        distance_is_too_close = True
        object_name, bounding_box = get_objects()
        cv2.rectangle(*bounding_box)

        write_and_print(
            "STOP: There is a {} less than 2 feet away, please turn to continue navigation.".format(
                object_name if object_name is not "" else "object"
            )
        )

    line = cv2.HoughLinesP(fin, 2, np.pi / 180, 20, 7, 7)
    if not (line is None):
        for i in line:
            cv2.line(screen, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (255, 0, 0), 10)

    l1dataset = []
    l2dataset = []
    try:
        straightxcors, straightycors = averageLanes(line)
        xcors, ycors = getPoints(line)

        l1dataset.append(straightxcors[0])
        l1dataset.append(straightycors[0])
        l2dataset.append(straightxcors[1])
        l2dataset.append(straightxcors[1])
        allstraightxcors = straightxcors[0] + straightxcors[1]
        allstraightycors = straightycors[0] + straightycors[1]

        l1m, l1b = linearRegression(l1dataset[0], l1dataset[1])
        l2m, l2b = linearRegression(l2dataset[0], l2dataset[1])

        allm, allb = linearRegression(allstraightxcors, allstraightycors)
        allxcor1 = int((allm * 350) + allb)
        allxcor2 = int(allb)

        filterl1x = []
        filterl1y = []
        filterl2x = []
        filterl2y = []

        for count, i in enumerate(ycors):
            if i * l2m + l2b < xcors[count]:
                filterl2x.append(xcors[count])
                filterl2y.append(i)
            else:
                filterl1x.append(xcors[count])
                filterl1y.append(i)

        l1inx1 = int((600 - l1b) / l1m)
        l1inx2 = int((350 - l1b) / l1m)

        l2inx1 = int((600 - l2b) / l2m)
        l2inx2 = int((350 - l2b) / l2m)

        cv2.line(screen, (int(l1inx1), 600), (int(l1inx2), 350), (0, 0, 0), 10)
        cv2.line(screen, (int(l2inx1), 600), (int(l2inx2), 350), (0, 0, 0), 10)

        # cv2.line(screen, (allxcor1, 600), (allxcor2,350), (255,0,0), 10)
        turning = ""

        results = intersection([l1m, l1b], [l2m, l2b])
        if not distance_is_too_close:
            if not (results is None):
                if results[0] > 400:
                    write_and_print("Turn Left")
                else:
                    write_and_print("Turn Right")
            else:
                write_and_print("Go straight")
        try:
            equ1, polyx1, polyy1 = polyReg(filterl2x, filterl2y)

            for i in range(len(polyx1)):
                if i == 0:
                    pass
                else:
                    cv2.line(
                        screen,
                        (int(polyx1[i]), int(polyy1[i])),
                        (int(polyx1[i - 1]), int(polyy1[i - 1])),
                        (255, 255, 0),
                        10,
                    )
        except Exception as e:
            print(e)
        try:
            equ2, polyx2, polyy2 = polyReg(filterl1x, filterl1y)

            for i in range(len(polyx2)):
                if i == 0:
                    pass
                else:
                    cv2.line(
                        screen,
                        (int(polyx2[i]), int(polyy2[i])),
                        (int(polyx2[i - 1]), int(polyy2[i - 1])),
                        (255, 255, 0),
                        10,
                    )

        except:
            pass

    except Exception as e:
        with open("write.txt", "w") as f:
            f.write("")

    return screen


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cap
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
### Copied from Author: Evan Juras
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
    default="1280x720",
)
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split("x")
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
### Copied from Author: Evan Juras

pkg = importlib.util.find_spec("tflite_runtime")
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if GRAPH_NAME == "detect.tflite":
        GRAPH_NAME = "edgetpu.tflite"

    # Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == "???":
    del labels[0]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT,
        experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
    )
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = input_details[0]["dtype"] == np.float32

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
print("Video Stream Started...Sleeping")
time.sleep(1)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
def get_objects():

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]["index"])[
        0
    ]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]["index"])[
        0
    ]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    i = np.argmax(scores)
    if scores[i] > min_conf_threshold:
        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1, (boxes[i][0] * imH)))
        xmin = int(max(1, (boxes[i][1] * imW)))
        ymax = int(min(imH, (boxes[i][2] * imH)))
        xmax = int(min(imW, (boxes[i][3] * imW)))

        bounding_box = (frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        # Draw label
        object_name = labels[
            int(classes[i])
        ]  # Look up object name from "labels" array using class index
        label = "%s: %d%%" % (
            object_name,
            int(scores[i] * 100),
        )  # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )  # Get font size
        label_ymin = max(
            ymin, labelSize[1] + 10
        )  # Make sure not to draw label too close to top of window

        xcenter = xmin + (int(round((xmax - xmin) / 2)))
        ycenter = ymin + (int(round((ymax - ymin) / 2)))
        # Print info
        print(
            "Object "
            + str(i)
            + ": "
            + object_name
            + " at ("
            + str(xcenter)
            + ", "
            + str(ycenter)
            + ")"
        )

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    return object_name, bounding_box


ToF.start_ranging()
# Running infinite loop to get constant video feeds
while True:
    try:
        _, screen = cap.read()

        screen = cv2.resize(screen, (800, 600))
    except:
        write_and_print("Done")
        ToF.stop_ranging()

    cv2.imshow("Navigation View", run(screen))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        ToF.stop_ranging()

cap.release()
# Clean up
cv2.destroyAllWindows()
videostream.stop()
