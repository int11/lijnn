import json
import lijnn
from lijnn.transforms import *
from model import Fast_R_CNN
from utils import SelectiveSearch
import numpy as np
from lijnn.datasets import VOCclassfication, VOCDetection
import os 
from lijnn import utils

class VOCSelectiveSearch(VOCDetection):
    def __init__(self, train=True, year=2007):
        super(VOCSelectiveSearch, self).__init__(train, year)
        self.order['background'] = 20
        annotationsdirName = 'SelectiveSearch'
        annotationsdir = os.path.join(os.path.dirname(self.annotationsdir), annotationsdirName)
        if not os.path.exists(annotationsdir):
            os.makedirs(annotationsdir, exist_ok=True)
            orderReverse = {v:k for k,v in self.order.items()}
            
            for i in range(super().__len__()):
                data = super().getitem(i)
                img, labels, bboxs = data['img'], data['labels'], data['bboxs']
                ssbboxs = SelectiveSearch(img)[:2000]
                jsondir = os.path.join(self.annotationsdir, self.nameindex[i] + '.json')
                with open(jsondir, 'r') as f:
                    jsondata = json.load(f)

                jsondata[annotationsdirName] = []
                for ssbbox in ssbboxs:
                    xmin, ymin, xmax, ymax = ssbbox
                    iouCandidate= [utils.IOU(ssbbox, bbox) for bbox in bboxs]
                    iouIndex = np.argmax(iouCandidate)
                    iou = iouCandidate[iouIndex]
                    label =  orderReverse[labels[iouIndex]] if iou > 0.5 else 'background'

                    temp = {"label": label, "iou": float(iou), "bndbox": {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)}} 
                    jsondata[annotationsdirName].append(temp)

                with open(os.path.join(annotationsdir, self.nameindex[i] + '.json'), 'w') as f:
                    json.dump(jsondata, f, indent=4)
        self.scan(annotationsdirName="SelectiveSearch")
        
    def getAnnotations1(self, index):
        annotations = super().getAnnotations(index)
        with open(os.path.join(self.annotationsdir, self.nameindex[index] + '.json'), 'r') as f:
            jsondata = json.load(f)

        annotations['ious'] = [1] * len(annotations['labels'])
        for i in jsondata['SelectiveSearch']:
            budbox = i['bndbox']
            annotations['bboxs'].append([budbox['xmin'], budbox['ymin'], budbox['xmax'], budbox['ymax']])
            annotations['labels'].append(self.order[i['label']])
            annotations['ious'].append(i['iou'])
            
        return annotations

    def getitem(self, index):
        annotations = self.getAnnotations1(index)
        annotations['img'] = self.getImg(index)
        return annotations

if __name__ == '__main__':
    dataset = VOCSelectiveSearch()
    for i in [2425, 2966, 4020, 4699]:
        print(dataset.nameindex[i])

