import numpy as np 
from skimage.feature import hog, canny
import cv2
from sklearn.svm import SVC
import random
import os
import cPickle

imgPath = '/home/dzy/local/traffic_Signs/Img/%05d.jpg'
annoPath = '/home/dzy/local/traffic_Signs/Anno/%05d.xml'

def getAnno(filePath):
    from xml.dom import minidom
    def getVal(x, tag):
        return x.getElementsByTagName(tag)[0].firstChild.data
    objs = minidom.parse(filePath).getElementsByTagName('object')
    objectList = []
    for o in objs:
        tmp = []
        coords = o.getElementsByTagName('bndbox')[0]
        for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
            tmp.append(int(getVal(coords, tag)))
        objectList.append(tmp)
    return objectList

# collect the positive samples and negative samples
# clf = SVC()
# X = []
# Y = []
# for i in range(10,2000,20):
#     img = cv2.imread(imgPath%i)
#     for j in range(4):
#         W,H = img.shape[:2]
#         x = random.randint(0,W-400)
#         y = random.randint(0,H-400)
#         w = random.randint(100, 400)
#         h = random.randint(100, 400)
#         roi = img[x:x+w,y:y+h,:]
#         cv2.imwrite('train/neg/'+str(i)+'_'+str(j)+'.jpg', roi)
#     try:
#         for idx,rect in enumerate(getAnno(annoPath%i)):
#             if rect[3]-rect[1] < 80 and rect[2]-rect[0]<80: continue
#             roi = img[rect[1]:rect[3],rect[0]:rect[2],:]
#             # cv2.imshow('123', roi)
#             # cv2.waitKey(0)
#             cv2.imwrite('train/pos/'+str(i)+'_'+str(idx)+'.jpg', roi)
#     except:
#         continue
def patch(roi):
    w,h = roi.shape
    ratio = float(w)/h
    Wx = int(max(1,0.5/ratio)+0.5)
    Hx = int(max(2*ratio,1)+0.5)
    # print w,h,Wx, Hx, ratio
    output = np.zeros((w*Wx, h*Hx),dtype=np.uint8)
    for xi in range(Wx):
        for yi in range(Hx):
            output[xi*w:(xi+1)*w,yi*h:(yi+1)*h] = roi[:,:]
    # print w,h, output.shape
    return output

def train():
    X = []
    Y = []
    clf=SVC()
    shape = (80, 40)
    path = 'train/'
    for img in os.listdir(path+'pos/'):
        # print img
        im0 = cv2.imread(path+'pos/'+img, 0)
        # print im0.shape
        # im0 = patch(im0)
        # cv2.imshow('123', im0)
        # cv2.waitKey(0)

        im = cv2.resize(im0, shape)
        (feature, h) = hog(im, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
        # feature = canny(im)
        # out = np.zeros_like(feature,dtype=np.uint8)
        # print out.shape
        # out[feature] = 255
        # print h.shape, im.shape, feature.shape
        X.append(feature)
        Y.append(1)
    print '123123123'
    
    for img in os.listdir(path+'neg/'):
        im0 = cv2.imread(path+'neg/'+img, 0)
        im = cv2.resize(im0, shape)
        # feature = canny(im)
        (feature, h) = hog(im, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
        X.append(feature)
        Y.append(0)
    print X[0].shape
    X = np.array(X)
    Y = np.array(Y)
    clf.fit(X,Y)
    cPickle.dump(clf,open('svm.clf','wb'))
    return clf

train()