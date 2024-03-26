
class VOC_SelectiveSearch(VOCclassfication):
    def __init__(self, train=True, year=2007, around_context=True):
        super(VOC_SelectiveSearch, self).__init__(train, year)
        self.around_context = around_context
        loaded = datasets.load_cache_npz(f'VOC_SelectiveSearch{year}', train=train)

        if loaded is not None:
            self.count, self.iou, self.g = loaded
        else:
            self.iou = [1.] * len(self.count)
            self.g = [*self.count[:, 1:5]]
            for i in range(VOCDetection.__len__(self)):
                img, labels, bboxs = VOCDetection.__getitem__(self, i)
                ssbboxs = utils.SelectiveSearch(img)[:2000]
                temp = []
                for ssbbox in ssbboxs:
                    bb_iou = [utils.IOU(ssbbox, bbox) for bbox in bboxs]
                    indexM = np.argmax(bb_iou)
                    temp.append(labels[indexM] if bb_iou[indexM] > 0.5 else 20)
                    self.iou.append(bb_iou[indexM])
                    self.g.append(bboxs[indexM])

                temp = np.append(ssbboxs, np.array(temp).reshape(-1, 1), axis=1)
                temp = np.pad(temp, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                self.count = np.append(self.count, temp, axis=0)
            self.iou, self.g = np.array(self.iou), np.array(self.g)

            # sort_index = np.apply_along_axis(lambda x: x[0], axis=1, arr=self.count).argsort()
            sort_index = np.empty(0, dtype=np.int32)
            for i in range(VOCDetection.__len__(self)):
                index = np.where(self.count[:, 0] == i)[0]
                sort_index = np.append(sort_index, index)

            self.count = self.count[sort_index]
            self.iou = self.iou[sort_index]
            self.g = self.g[sort_index]
            datasets.save_cache_npz({'label': self.count, 'iou': self.iou}, f'VOC_SelectiveSearch{year}', train=train)

    def __getitem__(self, index):
        temp = self.count[index]
        index, bbox, label = temp[0], temp[1:5], temp[5]
        img = self.getImg(index)
        img = AroundContext(img, bbox, 16) if self.around_context else img[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

        return img, label

    @staticmethod
    def labels():
        labels = VOCclassfication.labels()
        labels[20] = 'background'
        return labels
    



class VOC_fastrcnn(rcnn.VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007,
                 img_transform=compose([resize(224), toFloat(), z_score_normalize(Fast_R_CNN.mean, 1)]),
                 bbox_transform=bbox_transpose(224)):
        super(VOC_fastrcnn, self).__init__(train, year)
        self.add_transforms('img', img_transform)
        self.bbox_transform = bbox_transform

    def __getitem__(self, index):
        img = self.getImg(index)
        index = np.where(self.count[:, 0] == index)

        bbox, g = self.count[index][:, 1:], self.g[index]

        if self.bbox_transform is not None:
            bbox[:, :4] = self.bbox_transform(img.shape, bbox[:, :4])
            g = self.bbox_transform(img.shape, g)
        return img, bbox, self.iou[index].astype(np.float32), g

    def __len__(self):
        return super(lijnn.datasets.VOCclassfication, self).__len__()

