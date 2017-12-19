import numpy as np
from matplotlib import pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
import chainercv

"""
forked from https://github.com/leetenki/YOLOtiny_v2_chainer 
"""

class YOLOv2_tiny(chainer.Chain):

    def __init__(self, n_classes=20, n_boxes=5):
        super(YOLOv2_tiny, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(None, 64, 3, stride=1, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=1, pad=1, nobias=True)
            self.bn4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1, nobias=True)
            self.bn5 = L.BatchNormalization(256)
            self.conv6 = L.Convolution2D(None, 512, 3, stride=1, pad=1, nobias=True)
            self.bn6 = L.BatchNormalization(512)
            self.conv7 = L.Convolution2D(None, 1024, 3, stride=1, pad=1, nobias=True)
            self.bn7 = L.BatchNormalization(1024)
            self.conv8 = L.Convolution2D(None, 1024, 3, stride=1, pad=1, nobias=True)
            self.bn8 = L.BatchNormalization(1024)
            self.conv9 = L.Convolution2D(None, 125, 1, stride=1, pad=0, nobias=True)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)),slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn2(self.conv2(h)),slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn3(self.conv3(h)),slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn4(self.conv4(h)),slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn5(self.conv5(h)),slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn6(self.conv6(h)),slope=0.1)
        h = F.leaky_relu(self.bn7(self.conv7(h)),slope=0.1)
        h = F.leaky_relu(self.bn8(self.conv8(h)),slope=0.1)
        h = self.conv9(h)
        return h

def load_npz(filename, model):
    with np.load(filename) as f:
        weights = dict(f.items())

    for i in range(1, 9):
        exec('model.conv{0}.W.data = weights["c{0}/c/W"]'.format(i))
        exec('model.bn{0}.avg_mean = weights["c{0}/n/avg_mean"]'.format(i))
        exec('model.bn{0}.avg_var = weights["c{0}/n/avg_var"]'.format(i))
        exec('model.bn{0}.N = weights["c{0}/n/N"]'.format(i))
        exec('model.bn{0}.gamma.data = weights["c{0}/n/gamma"]'.format(i))
        exec('model.bn{0}.beta.data = weights["c{0}/b/b"]'.format(i))
    i = 9
    exec('model.conv{0}.W.data = weights["c{0}/c/W"]'.format(i))

def _sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)

def _softmax(x):
    x = np.array([x]) # reshape (len(x),) to (1, len(x))
    return F.softmax(x).data

def _get_detected_boxes(ans, n_grid_x, n_grid_y, n_bbox, n_classes, prob_thresh, img_width, img_height, biases):
    detected_boxes = []
    grid_width = img_width / float(n_grid_x)
    grid_height = img_height / float(n_grid_y)

    boxes = []
    scores = []
    labels = []
    for grid_y in range(n_grid_y):
        for grid_x in range(n_grid_x):
            for i in range(n_bbox):
                box = ans[grid_y, grid_x, i, 0:4] # (4,)
                conf = _sigmoid(ans[grid_y, grid_x, i, 4]) 
                probs = _softmax(ans[grid_y, grid_x, i, 5:])[0] # (20,)

                p_class = probs * conf # (20,)
                if np.max(p_class) < prob_thresh:
                    continue

                class_id = np.argmax(p_class)
                x = (grid_x + _sigmoid(box[0])) * grid_width
                y = (grid_y + _sigmoid(box[1])) * grid_height
                w = np.exp(box[2]) * biases[i][0] * grid_width
                h = np.exp(box[3]) * biases[i][1] * grid_height
                
                b = [y - 0.5*h, x - 0.5*w, y + 0.5*h, x+0.5*w]

                boxes.append(b)
                labels.append(class_id)
                scores.append(max(p_class))

    return boxes, labels, scores

def postprocess(ans, im_org_):
    img_width, img_height = 352,352
    org_img_height, org_img_width = im_org_.shape[0:2]
    n_grid_x = img_width//32
    n_grid_y = img_height//32
    n_classes = 20
    n_bbox = 5
    biases = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
    prob_thresh = 0.2
    iou_thresh = 0.04
    #org_img_height, org_img_width = im_org.shape[0:2]
    org_img_height, org_img_width = 352, 352
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train","tvmonitor"] 

    ans = ans.transpose(1, 2, 0) # (13, 13, 125)
    ans = ans.reshape(n_grid_y, n_grid_x, n_bbox, (n_classes + 5)) # (13, 13, 5, 25)

    # compute detected boxes
    bboxes, labels, scores = _get_detected_boxes(ans, n_grid_x, n_grid_y, n_bbox, n_classes, prob_thresh, org_img_width, org_img_height, biases)

    bboxes = np.asarray(bboxes, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    #for box, label, score in zip(bboxes, labels, scores):
    #    print(classes[label], score, box)
    #print()
    idx = chainercv.utils.non_maximum_suppression(bboxes, iou_thresh, score=scores, limit=None)
    bboxes, labels, scores = bboxes[idx], labels[idx], scores[idx]

    for box, label, score in zip(bboxes, labels, scores):
        print(classes[label], score, box)

    chainercv.visualizations.vis_bbox(im_org_, bboxes, label=labels, score=scores, label_names=classes, ax=None)
