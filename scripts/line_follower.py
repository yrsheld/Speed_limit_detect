#!/usr/bin/env python
import numpy as np
import rospy
import cv2
import imutils
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

class lineDetect:
    def __init__(self, camera_half_width, visualize, use_speed_limit):
        self.bridge = CvBridge()
        self.visualize = visualize
        self.use_speed_limit = use_speed_limit

        self.start = False

        # motion control parameters
        self.linear_speed = 1.0
        self.linear_multiplier = 1.0
        self.camera_half_width = camera_half_width
        self.angle_multiplier = 0.01
        self.decay = 0.8
        self.diff = 0.0

        rospy.init_node('speed_detect', anonymous=True)
        self.img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)
        
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        if self.visualize:
            self.img_pub = rospy.Publisher('/image_line', Image, queue_size=5)

        if self.use_speed_limit:
            self.speed_sub = rospy.Subscriber('/speed_limit', Float32, self.speed_cb)

    def speed_cb(self, speed_limit):
        self.linear_multiplier = speed_limit.data/100.0
        rospy.loginfo('Speed limit: %s'%(self.linear_speed * self.linear_multiplier))

    def img_cb(self, msg):
        '''
        follow yellow line
        '''
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Gaussian blur
        img_gauss = cv2.GaussianBlur(img, (5,5), 0)

        # HSV thresholding
        img_hsv = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2HSV)

        # yellow lower
        yellow_lower = np.array([20, 100, 100])
        yellow_higher = np.array([30, 255,255])
        yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_higher)

        # find line contour in masked image
        cnts = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) == 0:
            if self.start:
                # reach the end of the path, stop
                self.stop()
            return
        
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        if cv2.contourArea(cnt) < 0.5:
            return
        
        # get the mass center of line
        m = cv2.moments(cnt)
        cx = m['m10']/(m['m00']+1e-5)
        cy = m['m01']/(m['m00']+1e-5)

        if self.visualize:
            # draw contour
            cv2.drawContours(img, [cnt], 0, (255,0,0), 3)
            cv2.circle(img, (int(round(cx)), int(round(cy))), 4, (0,255,0), -1)
            # publish image result
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))

        self.calc_velocity(cx)
    
    def calc_velocity(self, centroid_x):
        '''
        Adjust linear velocity based on detected speed limit
        Adjust angular velocity based on line centroid
        '''
        m = Twist()
        self.diff = self.diff * self.decay + (centroid_x - self.camera_half_width) * (1-self.decay)
        m.angular.z =  - self.angle_multiplier * self.diff
        
        m.linear.x = self.linear_speed * self.linear_multiplier

        self.vel_pub.publish(m)
        self.start = True

    def stop(self):
        m = Twist()
        self.vel_pub.publish(m)
    

if __name__=="__main__":
    try:
        camera_half_width = rospy.get_param('camera_half_width', 320)
        visualize = rospy.get_param('visualize_line_detection', False)
        use_speed_limit = rospy.get_param('use_speed_limit', False)
        lineDetect(camera_half_width, visualize=True, use_speed_limit=True)

        while not rospy.is_shutdown():
            rospy.spin()

    except rospy.ROSInterruptException as e:
        print(e)