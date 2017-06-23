# This doesn't matter unless you have the actual .bag files, which aren't given in the git repo since each is multiple gigabytes.

import sys
import json
import rosbag

# import numpy as np
# from cv_bridge import CvBridge
# import cv2

def save_file(name, data):
    with open('{}.json'.format(name), 'wb') as f:
        json.dump(data, f)

def parse_ackermann(ackermann):
    d = ackermann.drive
    return d.steering_angle

bag = rosbag.Bag(sys.argv[1])
laser = {}
ackermann = {}
camera = {}

TOPICS = {
    '/sensors/scan': 'laser',
    # '/sensors/usb_cam/image_raw': 'camera',
    '/vesc/ackermann_cmd_mux/input/teleop': 'drive',
}

# print 'reading bag'
camera_messages = 0
total_camera_messages = 712
# bridge = CvBridge()

# def auto_canny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     blur_size = 3
#     image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
#     # v = np.median(image)
#     v = np.mean(image)
 
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)
 
#     # return the edged image
#     return edged

last_image = None
for topic, msg, t in bag.read_messages(topics=TOPICS.keys()):
    seconds = t.secs
    if TOPICS[topic] == 'laser':
        if seconds in laser:
            laser[seconds].append(msg.ranges + msg.intensities)
        else:
            laser[seconds] = [msg.ranges + msg.intensities]
    if TOPICS[topic] == 'camera':
        camera_messages += 1
        imgdata = bridge.imgmsg_to_cv2(msg)
        imgdata = cv2.resize(imgdata[:,:,0], (0, 0), fx=0.25, fy=0.25)
        if seconds not in camera:
            camera[seconds] = [imgdata.flatten().tolist()]
            if last_image is not None:
                camera[seconds-1].append(last_image.flatten().tolist())
                last_image = None
            print float(camera_messages)/float(total_camera_messages)
        else:
            last_image = imgdata
    if TOPICS[topic] == 'drive':
        if seconds in ackermann:
            ackermann[seconds].append(parse_ackermann(msg))
        else:
            ackermann[seconds] = [parse_ackermann(msg)]

bag.close()
print 'bag closed'

save_file('dict_data/laser', laser)
save_file('dict_data/ackermann', ackermann)
save_file('dict_data/camera', camera)
