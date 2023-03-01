import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import lijnn.utils
from example.CNN import *
from lijnn.transforms import *

N_CLASS = 20


class SlowROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.spatial_scale = spatial_scale
        self.size = output_size

    def forward(self, images, rois):
        rois[:, [1, 2]] = np.floor(rois[:, [1, 2]] * self.spatial_scale)
        rois[:, [3, 4]] = np.ceil(rois[:, [3, 4]] * self.spatial_scale)
        rois = rois.astype(int)

        n = rois.shape[0]
        x1 = rois[:, 1]
        y1 = rois[:, 2]
        x2 = rois[:, 3]
        y2 = rois[:, 4]

        res = []
        for i in range(n):
            img = images[int(rois[i][0])].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16_bn(pretrained=True)
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        # self.roipool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.roipool = SlowROIPool(output_size=(7, 7), spatial_scale=1/16)
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])

        _x = Variable(torch.Tensor(1, 3, 224, 224))
        _r = np.array([[0, 0., 0., 1., 1.]])
        _x = self.feature(self.roipool(self.seq(_x), _r).view(1, -1))
        feature_dim = _x.size(1)
        self.cls_score = nn.Linear(feature_dim, N_CLASS + 1)
        self.bbox = nn.Linear(feature_dim, 4 * (N_CLASS + 1))

        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois):
        res = inp
        res = self.seq(res)
        res = self.roipool(res, rois)
        res = res.detach()
        res = res.view(res.size(0), -1)
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, N_CLASS + 1, 4)
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc


path = os.path.join(lijnn.utils.cache_dir, "Fast_R_CNN")

start_epoch = 4
model = RCNN().cuda() if cuda.gpu_enable else RCNN()
model.load_state_dict(torch.load(os.path.join(path, f"{start_epoch}.pt")))

dataset = fastrcnn.VOC_fastrcnn(img_transform=compose(
    [resize(224), toFloat(), z_score_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
train_loader = fastrcnn.Hierarchical_Sampling(dataset=dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(start_epoch + 1, 100):
    sum_loss, sum_acc = 0, {}
    for x, t in train_loader:
        img, rois = torch.from_numpy(x[0]), x[1]
        t_label, p, g, u = t
        t = torch.from_numpy(t_label.astype(np.int64))
        t_bbox = torch.from_numpy(fastrcnn.getT_from_P_G(p, g))
        if cuda.gpu_enable:
            img = img.cuda()
            t = t.cuda()
            t_bbox = t_bbox.cuda()
        y, y_bbox = model(img, rois)

        loss, loss_sc, loss_loc = model.calc_loss(y, y_bbox, t, t_bbox)
        acc = fastrcnn.Faccuracy(y.cpu().detach().numpy(), y_bbox.cpu().detach().numpy(), t_label, p, g, u) \
            if cuda.gpu_enable else fastrcnn.Faccuracy(y.detach().numpy(), y_bbox.detach().numpy(), t_label, p, g, u)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data
        for key, value in acc.items():
            if not key in sum_acc:
                sum_acc[key] = value.data
            else:
                sum_acc[key] += value.data
        text = f"loss : {loss} {loss_sc} {loss_loc}"

        for key, value in acc.items():
            text += f" {key} : {value.data}"
        print(text)

    print(f"epoch {i}")
    text = f'train loss : {sum_loss / train_loader.max_iter}'
    for key, value in sum_acc.items():
        text += f'{key} : {value / train_loader.max_iter}'
    print(text)

    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, f"{i}.pt"))