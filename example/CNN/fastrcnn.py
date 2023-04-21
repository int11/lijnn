import lijnn.datasets
from lijnn import *
from lijnn.cuda import *
from lijnn import layers as L
from lijnn import functions as F
from lijnn.transforms import *
from example.CNN import VGG16, rcnn
import torch
import torch.nn as nn

class ROIPooling2D(Function):
    def __init__(self, output_size, spatial_scale):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
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
        self.roipool = SlowROIPool(output_size=(7, 7), spatial_scale=1/16)


    def forward(self, x, bboxs):
        assert bboxs.shape[-1] == 5, "bboxs input "
        self.x1 = torch.from_numpy(as_numpy(x))
        self.x1.requires_grad=True
        self.a = self.roipool(self.x1, as_numpy(bboxs.copy()))

        x = self.forward1(x, bboxs)
        
        print("forward", np.array_equal(as_numpy(x), self.a.detach().numpy()))

        return x 

    def forward1(self, x, bboxs):
        xp = cuda.get_array_module(x)
        assert bboxs.shape[-1] == 5, "bboxs input "
        
        bboxs = bboxs.copy()

        _, C, H, W = x.shape
        OH, OW = pair(self.output_size)
        N, _ = bboxs.shape

        y = xp.zeros((N, C, OH, OW), dtype=x.dtype)
        self.argmax_data = xp.zeros(y.shape, xp.int32)

        bboxs[:, [1, 2]] = xp.floor(bboxs[:, [1, 2]] * self.spatial_scale)
        bboxs[:, [3, 4]] = xp.ceil(bboxs[:, [3, 4]] * self.spatial_scale)


        #roi width, roi height, stridew, strideh, 
        a = xp.array([bboxs[:,3] - bboxs[:, 1], bboxs[:, 4] - bboxs[:, 2],  (bboxs[:, 3] - bboxs[:, 1])/OW, (bboxs[:,4] - bboxs[:, 2])/OH]).T
        L_sliceH = xp.tile(xp.arange(OH)[:, None], (N, 1, 2))
        #             xp.floor(_outh        * strideh)) + ymin
        L_sliceH[:, :, 0] = (xp.floor(L_sliceH[:, :, 0].T * a[:, 3]) + bboxs[:, 2]).T
        L_sliceH[:, :, 1] = (xp.ceil((L_sliceH[:, :, 1].T + 1) * a[:, 3]) + bboxs[:, 2]).T

        L_sliceW = xp.tile(xp.arange(OW)[:, None], (N, 1, 2))

        L_sliceW[:, :, 0] = (xp.floor(L_sliceW[:, :, 0].T * a[:, 2]) + bboxs[:, 1]).T
        L_sliceW[:, :, 1] = (xp.ceil((L_sliceW[:, :, 1].T + 1) * a[:, 2]) + bboxs[:, 1]).T 

        
        for n in range(N):
            for outh in range(OH):
                sliceh = L_sliceH[n][outh]
                for outw in range(OW):
                    slicew = L_sliceW[n][outw]
                    lenw = slicew[1] - slicew[0]

                    roi_data = x[int(bboxs[n][0]), :, sliceh[0]:sliceh[1], slicew[0]:slicew[1]].reshape(C, -1)
                    y[n, :, outh, outw] = xp.max(roi_data, axis=1)
                    index = xp.argmax(roi_data, axis=1)
                    ttt = (slicew[0] + sliceh[0] * W) + (index // lenw * W) + (index % lenw)
                    self.argmax_data[n, :, outh, outw] = ttt
        return y
        self._bottom_data_shape = x.shape

        OH, OW = pair(self.output_size)
        _, C, H, W = x.shape
        N, _ = bboxs.shape

        y = cuda.cupy.empty((N, C, OH, OW), dtype=x.dtype)
        self.argmax_data = cuda.cupy.empty(y.shape, np.int32)

        cuda.cupy.ElementwiseKernel(
            '''
            raw T x, T spatial_scale, int32 C,
            int32 H, int32 W, int32 pooled_height, int32 pooled_width,
            raw int32 bboxs
            ''',
            'T y, int32 argmax_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % C;
            int num = i / pooled_width / pooled_height / C;

            int roi_batch_ind = bboxs[num * 5 + 0];
            int roi_start_w = round(bboxs[num * 5 + 1] * spatial_scale);
            int roi_start_h = round(bboxs[num * 5 + 2] * spatial_scale);
            int roi_end_w = round(bboxs[num * 5 + 3] * spatial_scale);
            int roi_end_h = round(bboxs[num * 5 + 4] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);
            float bin_size_h = static_cast<float>(roi_height)
                           / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                           / static_cast<float>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                          * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                          * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                        * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), H);
            hend = min(max(hend + roi_start_h, 0), H);
            wstart = min(max(wstart + roi_start_w, 0), W);
            wend = min(max(wend + roi_start_w, 0), W);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (roi_batch_ind * C + c) * H * W;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * W + w;
                    if (x[data_offset + bottom_index] > maxval) {
                        maxval = x[data_offset + bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
            y = maxval;
            argmax_data = maxidx;
            ''', 'roi_pooling_2d_fwd'
        )(x, self.spatial_scale, C, H, W,
          OH, OW, bboxs, y,
          self.argmax_data)

        return y

    def backward(self, gy):
        asdf = self.a * torch.from_numpy(as_numpy(gy.data))
        adad = asdf.sum()
        adad.backward()
        x, bboxs = self.inputs
        gx, gbboxs = ROIPooling2DGrad(x.shape, self.argmax_data)(gy, bboxs)

        a = self.x1.grad.detach().numpy()
        b = as_numpy(gx.data)

        print("grad",(b[np.where(a != b)] - a[np.where(a != b)]).sum() < 0.0001)
        return gx, gbboxs


class ROIPooling2DGrad(Function):
    def __init__(self,  input_shape, argmax_data):
        self.input_shape = input_shape
        self.argmax_data = argmax_data

    def forward(self, gy, bboxs):
        xp = cuda.get_array_module(gy)

        _, C, H, W = self.input_shape
        N, _ = bboxs.shape

        gx = xp.zeros(self.input_shape, gy.dtype).ravel()
        a = bboxs[:,0] * C * H * W
        a = xp.broadcast_to(a.reshape(N,1,1,1), (N,C,1,1)) + (xp.arange(C) * H * W).reshape(1,C,1,1)
        a = self.argmax_data + a.reshape(N,C,1,1)
        gy_f = gy.ravel()

        np.add.at(gx, a, gy_f[np.arange(len(gy_f))])

        gx = gx.reshape(self.input_shape)

        return gx, None # gbboxs

    def backward(self, ggx, ggbboxs):
        # No trivial way to implement double-backward for this function.
        raise NotImplementedError


def roi_pooling1(x, rois, output_size, spatial_scale=1):
    return ROIPooling2D(output_size, spatial_scale)(x, rois)

class Fast_R_CNN(VGG16):
    def __init__(self, num_classes=21):
        super().__init__(imagenet_pretrained=True)
        self.fc8 = L.Linear(num_classes)
        self.conv8 = L.Conv2d_share_weight(num_classes, kernel_size=1, stride=1, pad=0, target=self.fc8)
        self._params.remove('conv8')
        self.Bbr = L.Linear(num_classes * 4)

    def forward(self, x, ssbboxs):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        # receptive field = 16
        # subsampling_ratio = 16
        x = roi_pooling1(x, ssbboxs, 7, 1/16)
        # x.shape = (N, 512, 7, 7)
        x = F.flatten(x)
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))

        cls_score = self.fc8(x)
        bbox_pred = self.Bbr(x)
        return cls_score, bbox_pred.reshape(len(x), -1, 4)


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


class VOC_fastrcnn(rcnn.VOC_SelectiveSearch):
    def __init__(self, train=True, year=2007,
                 img_transform=compose([resize(224), toFloat(), z_score_normalize(Fast_R_CNN.mean, 1)]),
                 bbox_transform=bbox_transpose(224)):
        super(VOC_fastrcnn, self).__init__(train, year, img_transform)
        self.bbox_transform = bbox_transform

    def __getitem__(self, index):
        img = self._get_index_img(index)
        index = np.where(self.count[:, 0] == index)

        bbox, g = self.count[index][:, 1:], self.g[index]

        if self.bbox_transform is not None:
            bbox[:, :4] = self.bbox_transform(img.shape, bbox[:, :4])
            g = self.bbox_transform(img.shape, g)
        return self.img_transform(img), bbox, self.iou[index].astype(np.float32), g

    def __len__(self):
        return super(lijnn.datasets.VOCclassfication, self).__len__()


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


class Hierarchical_Sampling(lijnn.iterator):
    def __init__(self, dataset=VOC_fastrcnn(), N=2, R=128, positive_sample_per=0.25, shuffle=True, gpu=False):
        super(Hierarchical_Sampling, self).__init__(dataset, N, shuffle, gpu)
        self.r_n = round(R / N)
        self.positive_sample_per = positive_sample_per

    def __next__(self):
        xp = cuda.cupy if self.gpu else np

        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        img_batch, ssbboxs = [], []
        label, p_batch, g_batch, u = [], [], [], []
        for i, (img, count, iou, g) in enumerate(batch):
            positive_img_num = int(self.r_n * self.positive_sample_per)

            POSindex = np.where(iou >= 0.6)[0]
            NEGindex = np.where(~(iou >= 0.6))[0]
            if self.shuffle:
                POSindex = POSindex[np.random.permutation(len(POSindex))]
                NEGindex = NEGindex[np.random.permutation(len(NEGindex))]
            POSindex = POSindex[:positive_img_num]
            NEGindex = NEGindex[:self.r_n - positive_img_num]

            index = np.concatenate((POSindex, NEGindex))

            p, g = count[index][:, :4], g[index]

            img_batch.append(img)
            ssbboxs.append(np.pad(p, ((0, 0), (1, 0)), mode='constant', constant_values=i))

            label.append(count[index][:, 4])
            p_batch.append(p)
            g_batch.append(g)
            u.append(np.ones_like(POSindex))
            u.append(np.zeros_like(NEGindex))

        self.iteration += 1
        return (xp.array(img_batch), xp.array(np.concatenate(ssbboxs))), \
               (xp.array(np.concatenate(label)), xp.array(np.concatenate(p_batch)), xp.array(np.concatenate(g_batch)),
                xp.array(np.concatenate(u)))


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
    optimizer = optimizers.Adam(alpha=0.0001)
    model.fit(epoch, optimizer, train_loader, loss_function=multi_loss, accuracy_function=Faccuracy,
              iteration_print=True, name=name)

if __name__ == "__main__":
	main_Fast_R_CNN()