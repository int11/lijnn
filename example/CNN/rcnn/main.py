import lijnn.datasets
from lijnn import *
from lijnn.cuda import *
from lijnn import layers as L
from lijnn import functions as F
from function import roi_pooling
from lijnn.transforms import *
from example.CNN import VGG16
from dataset import VOCSelectiveSearch
from model import Fast_R_CNN

class bbox_transpose:
    def __init__(self, img_outputsize):
        self.img_outputsize = img_outputsize

    def __call__(self, img_shape, bbox):
        H, W = img_shape[1:]
        OH, OW = pair(self.img_outputsize)
        bbox[:, 0] = np.floor(bbox[:, 0] * OW / W)
        bbox[:, 2] = np.ceil(bbox[:, 2] * OW / W)

        bbox[:, 1] = np.floor(bbox[:, 1] * OH / H)
        bbox[:, 3] = np.ceil(bbox[:, 3] * OH / H)

        return bbox


def xy1xy2_to_xywh(xy1xy2):
    xp = cuda.get_array_module(xy1xy2)
    xmin, ymin, xmax, ymax = xp.split(xy1xy2, 4, axis=1)
    xmin, ymin, xmax, ymax = xmin.ravel(), ymin.ravel(), xmax.ravel(), ymax.ravel()
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def getT_from_P_G(p, g):
    xp = cuda.get_array_module(p)
    p_x, p_y, p_w, p_h = xy1xy2_to_xywh(p)
    g_x, g_y, g_w, g_h = xy1xy2_to_xywh(g)
    return Variable(xp.array([(g_x - p_x) / p_w, (g_y - p_y) / p_h, np.log(g_w / p_w), np.log(g_h / p_h)], dtype=np.float32).T)

class Hierarchical_Sampling(lijnn.iterators.iterator):
    '''
    sampling total R samples from N images. it means get R/N samples from each image.
    positive sample that iou >= 0.5 is 25% from total sample , negative sample is iou < 0.5
    '''
    def __init__(self, dataset=VOCSelectiveSearch(), N=2, R=128, positive_sample_per=0.25, shuffle=True, gpu=False):
        super(Hierarchical_Sampling, self).__init__(dataset=dataset, batch_size=N, shuffle=shuffle, gpu=gpu)
        self.r_n = round(R / N)
        self.positive_sample_per = positive_sample_per
        self.positive_img_num = int(self.r_n * self.positive_sample_per)
        self.dataset.set_transforms('img', compose([resize(224), toFloat(), z_score_normalize(Fast_R_CNN.mean, 1)]))
        

    def next(self, batch_index):
        xp = cuda.cupy if self.gpu else np

        batchs = [self.dataset[i] for i in batch_index]
        img, labels, bboxs, ious = [], [], [], []

        for batch in batchs:
            POSindex = np.where(batch['ious'] >= 0.5)[0]
            NEGindex = np.where(~(batch['ious'] >= 0.5))[0]
            if self.shuffle:
                POSindex = POSindex[np.random.permutation(len(POSindex))]
                NEGindex = NEGindex[np.random.permutation(len(NEGindex))]
                
            POSindex = POSindex[:self.positive_img_num]
            NEGindex = NEGindex[:self.r_n - len(POSindex)]
            index = np.concatenate((POSindex, NEGindex))
            
            img.append(batch['img'])
            labels.append(batch['labels'][index])
            bboxs.append(batch['bboxs'][index])
            ious.append(batch['ious'][index])

        return (xp.array(xp.stack(img)), xp.array(xp.stack(bboxs))), (xp.array(xp.stack(labels)), xp.array(xp.stack(ious)))

def multi_loss(y, y_bbox, t_label, p, g, u):
    xp = cuda.get_array_module(y)
    loss_cls = F.softmax_cross_entropy(y, t_label)

    y_bbox = y_bbox[xp.arange(len(y)), t_label]
    t_bbox = getT_from_P_G(p, g)
    u = u[None].T
    loss_loc = F.smooth_l1_loss(y_bbox * u, t_bbox * u)

    return loss_cls + loss_loc


def Faccuracy(y, y_bbox, t_label, p, g, u):
    xp = cuda.get_array_module(y)
    y, y_bbox, t_label, p, g, u = as_numpy(y), as_numpy(y_bbox), as_numpy(t_label), as_numpy(p), as_numpy(g), as_numpy(u)

    # acc
    acc = (y.argmax(axis=1) == t_label).mean()
    # iou
    y_bbox = y_bbox[np.arange(len(y)), t_label]
    index = u.astype(np.bool_)
    y_bbox, p = y_bbox[index], p[index]
    p_x, p_y, p_w, p_h = xy1xy2_to_xywh(p)
    x = p_w * y_bbox[:, 0] + p_x
    y = p_h * y_bbox[:, 1] + p_y
    w = p_w * np.exp(y_bbox[:, 2])
    h = p_h * np.exp(y_bbox[:, 3])
    y_bbox[:, 0], y_bbox[:, 1], y_bbox[:, 2], y_bbox[:, 3] = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    iou = sum([utils.IOU(a, b) for a, b in zip(y_bbox, g)]) / len(y_bbox)
    return {'acc': Variable(xp.array(acc)), 'iou': Variable(xp.array(iou))}


def main_Fast_R_CNN(name='default'):
    epoch = 10

    train_loader = Hierarchical_Sampling()
    model = Fast_R_CNN()

    model.info(np.zeros((1, 3, 224, 224)), np.array([[0, 0, 100, 100]]))
    optimizer = optimizers.Adam(alpha=0.0001)
    model.fit(epoch, optimizer, train_loader, loss_function=multi_loss, accuracy_function=Faccuracy,
              iteration_print=True, name=name)

if __name__ == "__main__":
	main_Fast_R_CNN()