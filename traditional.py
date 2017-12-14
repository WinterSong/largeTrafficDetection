import cv2
import numpy as np 
from skimage.feature import hog, canny
from sklearn.svm import SVC
import cPickle as Pickle
import time

imgPath = '/home/dzy/local/traffic_Signs/Img/%05d.jpg'
annoPath = '/home/dzy/local/traffic_Signs/Anno/%05d.xml'
clfFile =open('svm.clf','rb') 
clf = Pickle.load(clfFile)
thres = 0
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

def ifin(res):
    newRes = []
    for i, anno in enumerate(res):
        flag = False
        for j, a in enumerate(res):
            if i == j: continue
            if (anno[0]>a[0] and anno[1]>a[1] and anno[2]<a[2] and anno[3]<a[3]):
                flag = True
                break
        if not flag:
            newRes.append(anno)
    return newRes

def IOU(p1, p2):
    # print p1,p2
    if min(p1[2],p2[2])<max(p1[0], p2[0]):
        return 0
    if min(p1[3],p2[3])<max(p1[1], p2[1]):
        return 0
    # print 'normal'
    overlap = (min(p1[2],p2[2])-max(p1[0], p2[0]))* (min(p1[3],p2[3])-max(p1[1], p2[1]))
    total = (p1[2]-p1[0])*(p1[3]-p1[1]) + (p2[2]-p2[0])*(p2[3]-p2[1])
    # print overlap,'/',total-overlap
    return float(overlap)/(total-overlap)

def find(rectList, p):
    # print 'find in'
    for idx, rect in enumerate(rectList):
        newRect = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
        overlap = IOU(p, newRect)
        # print 'overlap:',overlap
        if overlap>0.3:
            return idx
    return -1

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
        if w<thres or h<thres: continue
        roi = img[y:y+h,x:x+w,:]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # roi = patch(roi)
        # cv2.imshow('123', roi)
        # cv2.waitKey(0)
        # cv2.imshow('roi', roi)
        # cv2.waitKey(0)

        # tmp = np.zeros_like(roi)
        # cannyFeature = canny(roi)
        # tmp[cannyFeature] = 1
        # cnt = np.sum(tmp)
        # if cnt<100:
        #     continue
        # print 'canny True'

        im = cv2.resize(roi, (80,40))
        (feature, himg) = hog(im, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
        rate = clf.predict([feature])
        # print rate
        if clf.predict([feature]) == 0:
            # print "False"
            continue
        # print 'True'
        # feature = hog(im)
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
        rectList.append((x,y,w,h))
    # cv2.imshow('123', img)
    # rectList = ifin(rectList)
    detected = [0]*len(rectList)
    # np.array(rectList).tofile('123.raw')
    try:
        # print i
        for anno in getAnno('/home/dzy/local/traffic_Signs/Anno/%05d.xml'%i):
            if anno[2]-anno[0]<thres or anno[3]-anno[1]<thres:
                continue
            p = (anno[0], anno[1], anno[2], anno[3])
            res = find(rectList, p)
            # print '123',23
            if res < 0:
                miss += 1
            else:
                TruePos += 1
            # print 'res',res
            if res>=0:
                detected[res] += 1
            # else:
                # print 'bbox not found'
    except:
        # print 'no annotation file found!'
        return 0, len(rectList), 0

    for i in detected:
        if i == 0:
            FalsePos += 1
    return TruePos, FalsePos, miss

def main():
    T,F,M = 0,0,0
    start = time.time()
    for i in range(1000):
        t,f,m = DIP(i)
        # print t,f,m
        T+=t
        F+=f
        M+=m
        if i % 1000 == 0:
            print i
    print T,F,M
    print time.time()-start

def test():
    i = 856
    # image = cv2.imread('/home/dzy/local/traffic_Signs/Img/%05d.jpg'%i)
    DIP(i)
    # image =image[369:448,973:1209,:]
    # image = cv2.resize(image, (110,60))
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # out = hog(img)
    # print out.shape

if __name__ == '__main__':
    # main()
    test()

# thres:100 DIP: 479 5661 499
# thres:100 DIP+canny: 479 4592 472
# thres:0 DIP:764 6310 877 of 1000 71.7195930481
# thres:0 DIP+hog+SVM 341 287 1300 of 1000 1519.84161305
# thres:0 DIP+canny   957 2624 684 of 1000 98.9423570633
# thres:0 DIP+Canny remove inside: 820 1481 821 of 1000 97.9300642014