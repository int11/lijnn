import json
from utils import SelectiveSearch
import numpy as np
from lijnn.datasets import VOCclassfication, VOCDetection
import os 
from lijnn import utils
import xml.etree.ElementTree as ET

class VOC_SelectiveSearch(VOCDetection):
    def __init__(self, train=True, year=2007):
        super(VOC_SelectiveSearch, self).__init__(train, year)
        self.order['background'] = 20
        annotationsdirName = 'SelectiveSearch'
        annotationsdir = os.path.join(os.path.dirname(self.annotationsdir), annotationsdirName)
        # if not os.path.exists(annotationsdir):
        os.makedirs(annotationsdir, exist_ok=True)
        orderReverse = {v:k for k,v in self.order.items()}
        
        for i in range(super().__len__()):
            data = super().getitem(i)
            img, labels, bboxs = data['img'], data['labels'], data['bboxs']
            ssbboxs = SelectiveSearch(img)[:2000]
            jsondir = os.path.join(self.annotationsdir, self.nameindex[i] + '.json')
            with open(jsondir, 'r') as f:
                jsondata = json.load(f)['annotation']

            jsondata[annotationsdirName] = []
            for ssbbox in ssbboxs:
                xmin, ymin, xmax, ymax = ssbbox
                iouCandidate= [utils.IOU(ssbbox, bbox) for bbox in bboxs]
                iouIndex = np.argmax(iouCandidate)
                iou = iouCandidate[iouIndex]
                label =  orderReverse[labels[iouIndex]] if iou > 0.5 else 'background'

                temp = {"label": label, "iou": int(iou), "bndbox": {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)}} 
                jsondata[annotationsdirName].append(temp)

            with open(os.path.join(annotationsdir, self.nameindex[i] + '.json'), 'w') as f:
                json.dump(jsondata, f, indent=4)


if __name__ == '__main__':
    train_loader = VOC_SelectiveSearch()
