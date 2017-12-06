import cv2
import numpy as np 
from skimage.feature import hog, canny
from sklearn.svm import SVC
import cPickle as Pickle

imgPath = '/home/dzy/local/traffic_Signs/Img/%05d.jpg'
annoPath = '/home/dzy/local/traffic_Signs/Anno/%05d.xml'
clfFile =open('svm.clf','rb') 
clf = Pickle.load(clfFile)
thres = 20
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

def IOU(p1, p2):
    # print p1,p2
    if min(p1[2],p2[2])<max(p1[0], p2[0]):
        return 0
    if min(p1[3],p2[3])<max(p1[1], p2[1]):
        return 0
    overlap = (min(p1[2],p2[2])-max(p1[0], p2[0]))* (min(p1[3],p2[3])-max(p1[1], p2[1]))
    total = (p1[2]-p1[0])*(p1[3]-p1[1]) + (p2[2]-p2[0])*(p2[3]-p2[1])
    # print overlap,'/',total-overlap
    return float(overlap)/(total-overlap)

def find(rectList, p):
    for idx, rect in enumerate(rectList):
        newRect = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
        if IOU(p, newRect)>=0.5:
            return idx
    return -1

def DIP(i):
    img = cv2.imread('/home/dzy/local/traffic_Signs/Img/%05d.jpg'%i)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_1 = np.logical_and(img2[:,:,0]>77, img2[:,:,0]<125)
    mask_2 = np.logical_and(img2[:,:,1]>43, img2[:,:,2]>46)
    mask = np.logical_and(mask_1, mask_2)
    binary_img = np.zeros_like(mask, dtype=np.uint8)
    binary_img[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10,10))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN,kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    imgRes = img.copy()
    mask2 = np.logical_not(closed)
    imgRes[mask2] = (0,0,0)
    contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    rectList = []
    TruePos = 0
    FalsePos = 0
    miss = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print w,h
        if w<thres or h<thres: continue
        roi = img[y:y+h,x:x+w,:]
        # cv2.imshow('123', roi)
        # cv2.waitKey(0)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        tmp = np.zeros_like(roi)
        # feature = canny(roi)
        im = cv2.resize(roi, (80,40))
        (feature, h) = hog(im, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
        if clf.predict([feature]) == 0:
            # print "False"
            continue
        # print 'True'
        # tmp[feature] = 1
        # cnt = np.sum(tmp)
        # if cnt<100: continue
        # cv2.imshow('roi', tmp)
        # cv2.waitKey(0)
        # feature = hog(im)
        rectList.append((x,y,w,h))
    detected = [0]*len(rectList)
    # np.array(rectList).tofile('123.raw')
    try:
        for anno in getAnno('/home/dzy/local/traffic_Signs/Anno/%05d.xml'%i):
            if anno[2]-anno[0]<thres or anno[3]-anno[1]<thres:
                continue
            p = (anno[0], anno[1], anno[2], anno[3])
            res = find(rectList, p)
            if res < 0:
                miss += 1
            else:
                TruePos += 1
            detected[res] += 1
    except:
        return 0, len(rectList), 0

    for i in detected:
        if i == 0:
            FalsePos += 1
    return TruePos, FalsePos, miss

def main():
    T,F,M = 0,0,0
    for i in range(5000):
        t,f,m = DIP(i)
        T+=t
        F+=f
        M+=m
        if i % 1000 == 0:
            print i
    print T,F,M

def test():
    i = 856
    # image = cv2.imread('/home/dzy/local/traffic_Signs/Img/%05d.jpg'%i)
    DIP(i)
    # image =image[369:448,973:1209,:]
    # image = cv2.resize(image, (110,60))
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('123', image)
    # cv2.waitKey(0)
    # out = hog(img)
    # print out.shape

if __name__ == '__main__':
    main()
    # test()

# thres:100 DIP: 479 5661 499
# thres:100 DIP+canny: 479 4592 472