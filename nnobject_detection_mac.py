from __future__ import division
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import datetime
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2
import imutils
from object_detection.utils import label_map_util
import time
#from utils import label_map_util
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import visualization_utils as vis_util

from lib.mouse import Mouse
from lib.polygon import drawQuadrilateral
from lib.user_interaction import getPerpectiveCoordinates
from lib.fgbg_calculations import getThresholdedFrame
from lib.heatmap import Heatmap
from lib.coordinate_transform import windowToFieldCoordinates

mouse = Mouse()

ap = argparse.ArgumentParser()
ap.add_argument("-fw", "--fieldwidth", type=int, default=334, help="top-view field width")
ap.add_argument("-fh", "--fieldheight", type=int, default=280, help="top-view field height")
args = vars(ap.parse_args())
resultWidth = args["fieldwidth"]
resultHeight = args["fieldheight"]
padding = 20

resultWidth = args["fieldwidth"]
resultHeight = args["fieldheight"]
(xb1, yb1) = (padding, padding)
(xb2, yb2) = (padding + resultWidth, padding)
(xb3, yb3) = (padding + resultWidth, padding + resultHeight)
(xb4, yb4) = (padding, padding + resultHeight)
field = np.zeros((resultHeight + padding*2,resultWidth + padding*2,3), np.uint8)
drawQuadrilateral(field, [(xb1, yb1), (xb2, yb2), (xb3, yb3), (xb4, yb4)], 0, 255, 0, 2)
heatmap = Heatmap(field, resultWidth, resultHeight)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Note: Model used for SSDLite_Mobilenet_v2
PATH_TO_CKPT = '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
#PATH_TO_CKPT = '/home/muhammad/umer-env/lib/python3.6/site-packages/tensorflow/models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
#print(PATH_TO_CKPT)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
#PATH_TO_LABELS = '/home/muhammad/umer-env/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

def detect_team(image, show = False):
    # define the list of boundaries
    boundaries = [
    ([17, 15, 100], [50, 56, 200]), #red
    ([25, 146, 190], [96, 174, 250]) #yellow
    ]
    i = 0
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
#         print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return 'red'
        elif ratio > 0.01 and i == 1:
            return 'yellow'

        i += 1
        
        if show == True:
            cv2.imshow("images", np.hstack([image, output]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
              cv2.destroyAllWindows() 
    return 'not_sure'

## To View Color Mask
'''filename = 'image2.jpg'
image = cv2.imread(filename)
resize = cv2.resize(image, (640,360))
detect_team(resize, show=True)'''

#intializing the web camera device

out = cv2.VideoWriter('soccer_out2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,360))

filename = '/Users/muhammadumerfarooqi/Documents/FYP/futsal.mp4'
#filename = '/home/muhammad/umer-env/ausvsperu.mp4'
cap = cv2.VideoCapture(filename)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   counter = 0
   #i = 1
   while (True):
      ret, image_np = cap.read()
      counter += 1
      if ret:
          h = image_np.shape[0]
          w = image_np.shape[1]

      if not ret:
        break
      #image_np = imutils.resize(image_np, width=1200)
      if counter % 1 == 0:
          if counter == 1:
            coords2 = getPerpectiveCoordinates(image_np, 'frame', mouse)
            '''resultWidth = coords2[3][0] - coords2[0][0]
            resultHeight = coords2[1][1] - coords2[0][1]
            (xb1, yb1) = (padding, padding)
            (xb2, yb2) = (padding + resultWidth, padding)
            (xb3, yb3) = (padding + resultWidth, padding + resultHeight)
            (xb4, yb4) = (padding, padding + resultHeight)
            field = np.zeros((resultHeight + padding*2,resultWidth + padding*2,3), np.uint8)
            drawQuadrilateral(field, [(xb1, yb1), (xb2, yb2), (xb3, yb3), (xb4, yb4)], 0, 255, 0, 2)
            heatmap = Heatmap(field, resultWidth, resultHeight)'''
            print(coords2)

          #drawQuadrilateral(frame, coords2, 0, 255, 0, 2)



          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          #tf.Print(boxes)
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3,
              min_score_thresh=0.5)
        
          loc = {}
          
          for n in range(len(scores[0])):
             if scores[0][n] > 0.50:
                # Calculate position
                ymin = int(boxes[0][n][0] * h)
                xmin = int(boxes[0][n][1] * w)
                ymax = int(boxes[0][n][2] * h)
                xmax = int(boxes[0][n][3] * w)


                # Find label corresponding to that class
                for cat in categories:
                    if cat['id'] == classes[0][n]:
                        label = cat['name']

                ## extract every person
                if label == 'person':
                    #crop them
                    crop_img = image_np[ymin:ymax, xmin:xmax]
                    
                    #print(crop_img)
                    color = detect_team(crop_img)
                    if color != 'not_sure':
                      coords = (xmin, ymin)
                      if color == 'red':
                        #if int((xmin + xmax)/2-coords2[0][0]) > 0: 
                        midx = int((xmin + xmax)/2)#-coords2[0][0])
                        #else: minx = 0

                        #if int((ymin + ymax)/2-coords2[0][1]) > 0:
                        midy = int((ymin + ymax)/2)#-coords2[0][1])
                        #else: midy = 0

                        basePoint = (midx, midy) 
                        print(basePoint)
                        print("holycow")
                        (xbRel, ybRel) = heatmap.getPosRelativeCoordinates(basePoint, coords2)
                        print(xbRel, ybRel)
                        print("fuck")
                        #(xb, yb) = heatmap.getPosAbsoluteCoordinates((xbRel, ybRel), (xb1, yb1))
                        #print(xb,yb)
                        #print("bitch")
                        if counter % 5 == 0:
                          if counter < 200:
                            heatmap.drawOpacityCircle((xbRel, ybRel), 255, 0, 0, 0, 15)
                          else:
                            heatmap.drawOpacityCircle((xbRel, ybRel), 255, 0, 0, 0, 20)
                        loc[coords] = 'SPAIN'
                        '''if counter % 5 == 0:
                          heatmap.drawOpacityCircle(basePoint, 255, 0, 0, 0, 15)'''
                        
                      else:
                        loc[coords] = 'BRAZIL'
                        
        ## print color next to the person
          for key in loc.keys():
            text_pos = str(loc[key])
            cv2.putText(image_np, text_pos, (key[0], key[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2) # Text in black
      
      cv2.imshow('image', image_np)
      cv2.imshow('field',field)
      out.write(image_np)
      #
      if cv2.waitKey(10) & 0xFF == ord('q'):
          time.sleep(0)
          cv2.destroyAllWindows()
          cap.release()
          break
    #print(counter)
    #cv2.destroyWindow(image_np)
    #print("height:" + str(height))cap.release()
