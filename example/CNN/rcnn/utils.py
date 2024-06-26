from lijnn import cuda
import cv2 as cv

def SelectiveSearch(img, xywh=False):
    """
    Args:
        xywh (bool): if true return xywh format, else return xmin, ymin, xmax, ymax format
    """
    
    xp = cuda.get_array_module(img)
    img = cuda.as_numpy(img)

    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img[::-1].transpose(1, 2, 0))
    ss.switchToSelectiveSearchFast()
    ssbboxs = ss.process()
    if not xywh:
        ssbboxs[:, 2:4] = ssbboxs[:, 0:2] + ssbboxs[:, 2:4]
    return xp.array(ssbboxs)