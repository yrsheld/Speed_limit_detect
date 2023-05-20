#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class speedDetect:
    def __init__(self):
        self.bridge = CvBridge()
        self.detector = speedDetector()

        rospy.init_node('speed_detect', anonymous=True)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)
        self.pub = rospy.Publisher('/speed_limit', Float32, queue_size=1)
        self.img_pub = rospy.Publisher('/image_result', Image, queue_size=1)
    
    def img_cb(self, msg):
        '''
        find region of interests
        '''
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_res, speed = self.detector.feature_match(img_bgr)
        
        # If speed limit sign detected
        if speed > 0:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(img_res))
            self.pub.publish(speed)
    
class speedDetector:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.speeds = [16, 30, 40, 50, 110]
        
        # Load speed limit images and compute SIFT features
        self.temp_images = []
        self.temp_SIFT = []
        self.detect_init()
        
        # camera image and its SIFT features
        self.image = None
        self.image_SIFT = None
    
        # flann based matching between templates and images
        self.flann = self.flann_init()
        self.min_match_count = 18

    def detect_init(self):
        '''
        Precompute keypoints and descriptors for all speed limit signs (i.e., templates)
        '''
        template_dir = rospy.get_param('template_images_dir', '')
        
        for speed in self.speeds:
            filename = '%s/Speed_%s.png'%(template_dir, speed)
            print('Filename: ', filename)
            speed_img = cv2.imread(filename)
            speed_img_gray = cv2.cvtColor(speed_img, cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(speed_img_gray, None)
            self.temp_images.append(speed_img)
            self.temp_SIFT.append((keypoints, descriptors))

    def flann_init(self):
        '''
        Create flann-based matcher
        '''
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=100)

        return cv2.FlannBasedMatcher(index_params, search_params)

    def feature_match(self, img_input):
        '''
        Compute SIFT features of input image
        Match SIFT features with all templates
        '''
        self.image = img_input.copy()
        
        # Compute keypoints and descriptors in the img    
        img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        self.image_SIFT = self.sift.detectAndCompute(img_gray, None)
        key_input, desc_input = self.image_SIFT

        best_match_param = dict(match_count=0, temp_id=-1, matches=None, matchesMask=None)
        for i, (key_temp, desc_temp) in enumerate(self.temp_SIFT):
            matches = self.flann.knnMatch(desc_input, desc_temp, k=2)
            good_match = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_match.append(m)

            matchesMask = None
            if len(good_match) > self.min_match_count and len(good_match) > best_match_param['match_count']:
                # record current values
                best_match_param['match_count'] = len(good_match)
                best_match_param['temp_id'] = i
                best_match_param['matches'] = good_match
                src_pts = np.float32([key_input[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                dst_pts = np.float32([key_temp[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                best_match_param['matchesMask'] = mask.ravel().tolist()

        return self.calc_result(best_match_param)


    def calc_result(self, best_match_param):
        '''
        Return
        - image visualizing SIFT matching
        - speed sign prediction result
        '''
        if best_match_param['temp_id'] < 0:
            # no signs detected, return original image
            return self.image, -1
        else:
            # grab the sign with the largest number of good matches
            temp_id = best_match_param['temp_id']
            draw_params = dict(matchColor=(0,0,255), 
                                singlePointColor=None,
                                matchesMask=best_match_param['matchesMask'],
                                flags=2)
            img_result = cv2.drawMatches(self.image, self.image_SIFT[0], 
                                    self.temp_images[temp_id], self.temp_SIFT[temp_id][0],
                                    best_match_param['matches'], None, **draw_params)

            return img_result, self.speeds[temp_id]

if __name__=="__main__":
    try:
        speedDetect()

        while not rospy.is_shutdown():
            rospy.spin()

    except rospy.ROSInterruptException as e:
        print(e)
