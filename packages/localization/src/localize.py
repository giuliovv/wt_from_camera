#!/usr/bin/env python

"""
Read from camera and publish the car position.

From https://gist.github.com/waveform80/22dea34379d5a7171ce4
"""

import picamera
import picamera.array
import numpy as np
import cv2
import rospy

from localization.msg import DuckPose
from sensor_msgs.msg import CompressedImage

def get_car(img):
    """
    Extract the car from the image.

    :param img: image
    :return: front coord, left coord, theta
    """
    # scale_percent = 60
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_color1_blue = np.array([50, 200, 90])
    hsv_color2_blue = np.array([100, 300, 300])
    mask_blue = cv2.inRange(img_hsv, hsv_color1_blue, hsv_color2_blue)

    hsv_color1_pink = np.array([150, 50, 200])
    hsv_color2_pink = np.array([200, 100, 250])
    mask_pink = cv2.inRange(img_hsv, hsv_color1_pink, hsv_color2_pink)
    
    back_coo = np.argwhere(mask_pink==255).mean(axis=0)[::-1]
    front_coo = np.argwhere(mask_blue==255).mean(axis=0)[::-1]
    
    x_center = (front_coo[0] + back_coo[0])/2
    y_center = (front_coo[1] + back_coo[1])/2
    angle = np.arctan2(-front_coo[1]+back_coo[1], -front_coo[0]+back_coo[0])
    
    return x_center, y_center, angle

class MyAnalysis(picamera.array.PiRGBAnalysis):
    def __init__(self, camera, coordinates_pub, image_pub, rectify_alpha):
        super(MyAnalysis, self).__init__(camera)
        self._camera_parameters = None
        # TODO find usual value
        self._mapx, self._mapy = None, None
        self.rectify_alpha = rectify_alpha
        self.coordinates_pub = coordinates_pub
        self.image_pub = image_pub

    def analyse(self, image_np):
        # Rectify
        image_np = cv2.remap(image_np, self._mapx, self._mapy, cv2.INTER_NEAREST)
        # Img has origin on top left, after the interpolation it will be rotated of 90 degrees, need to prevent that
        image_np = cv2.flip(image_np, 0)

        localized = False

        try:
            x, y, theta = get_car(image_np)
            localized = True
        except ValueError:
            print("No lines found.")

        # Rotate and remove offset
        scale_x = rospy.get_param('scale_x', 0.005425407359412304)
        x = x*scale_x
        scale_y = rospy.get_param('scale_y', 0.0030901948655952406)
        y = y*scale_y
        offset_x = rospy.get_param('offset_x', 1.3492280373361594)
        x -= offset_x
        offset_y = rospy.get_param('offset_y', 0.8957448504963013)
        y -= offset_y

        # DuckPose
        pose = DuckPose()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "watchtower00/localization"
        if localized:
            pose.x = x
            pose.y = y
            pose.theta = theta
            pose.success = True
        else:
            pose.x = -1
            pose.y = -1
            pose.theta = -1
            pose.success = False

        self.coordinates_pub.publish(pose)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

def main():
    rospy.init_node("localize")
    coordinates_pub = rospy.Publisher("/watchtower00/localization", DuckPose, queue_size=1)
    image_pub = rospy.Publisher("/watchtower00/camera/image/compressed", CompressedImage, queue_size=1)
    rectify_alpha = rospy.get_param("~rectify_alpha", 0.0)
    with picamera.PiCamera() as camera:
        # TODO fix with correct resolution
        camera.resolution = (1080, 1350)
        camera.framerate = 10
        output = MyAnalysis(camera, coordinates_pub=coordinates_pub, image_pub=image_pub, rectify_alpha=rectify_alpha)
        camera.start_preview(output, 'rgb')
        print('FPS: %.2f' % (output.frame_num / 10))


if __name__ == "__main__":
    main()