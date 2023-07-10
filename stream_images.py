import os
import time

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node("image_streamer")

SOURCE = "/workspace/plate/"
cv_bridge = CvBridge()

publisher = rospy.Publisher("/camera/camera/image", Image)



for img_path in os.scandir(SOURCE):
    numpy_img = cv2.imread(img_path.path)
    ros_image = cv_bridge.cv2_to_imgmsg(numpy_img)
    publisher.publish(ros_image)
    time.sleep(5)
