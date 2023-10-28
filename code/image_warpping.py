import numpy as np
from numpy import array as arr
import math



def get_boundary(img, H):
    """
    This function calculate the new bounding box of the source image after the transformation by matrix H
    :param img: The source image in form [height, row, channel]
    :param H: The homography matrix
    :return: min. x, max. x, min. y, max. y
    """
    imgH, imgW = img.shape[0], img.shape[1]
    tl = arr([0, 0, 1])
    tr = arr([0, imgW-1, 1])
    bl = arr([imgH-1, 0, 1])
    br = arr([imgH-1, imgW-1, 1])
    tl = np.dot(H, tl)
    tr = np.dot(H, tr)
    bl = np.dot(H, bl)
    br = np.dot(H, br)
    x = arr([tl[0]/tl[2], tr[0]/tr[2], bl[0]/bl[2], br[0]/br[2]])
    y = arr([tl[1]/tl[2], tr[1]/tr[2], bl[1]/bl[2], br[1]/br[2]])
    return math.ceil(np.min(x)), math.ceil(np.max(x)), math.ceil(np.min(y)), math.ceil(np.max(y))


def image_warping(dst, src, h):
    """
    This function performs image wrapping by multiplying the inverse of homography matrix to points in the destination to get pixel value, and perform mean averaging if the inverse location contains floating point.
    :param dst: The destination image in shape [height, width, channel]
    :param src: The source image in shape [height, width, channel]
    :param h: The homographic matrix
    :return: A merged image
    """
    src_min_x, src_max_x, src_min_y, src_max_y = 0, src.shape[0], 0, src.shape[1]
    min_x, max_x, min_y, max_y = get_boundary(src, h)
    dst_max_x, dst_max_y, dst_min_x, dst_min_y = dst.shape[0], dst.shape[1], 0, 0
    res_min_x, res_max_x, res_min_y, res_max_y = min(min_x, dst_min_x), max(max_x, dst_max_x), min(min_y, dst_min_y), max(max_y, dst_max_y)
    res_h, res_w = int(math.ceil(res_max_x)-math.floor(res_min_x)), int(math.ceil(res_max_y)-math.floor(res_min_y))
    inv_h = np.linalg.inv(h)
    translation = arr([[1, 0, -res_min_x], [0, 1, -res_min_y], [0, 0, 1]])
    res = np.zeros(shape=[res_h, res_w, 3])
    for row in range(min_x, max_x):
        for col in range(min_y, max_y):
            target = np.dot(translation, arr([row, col, 1]))
            pixel_position = np.dot(inv_h, arr([row, col, 1]))
            x, y = pixel_position[0]/pixel_position[2], pixel_position[1]/pixel_position[2]
            try:
                res[target[0], target[1], :] = src[x, y, :]
            except IndexError:
                if 0 <= x < src_max_x and 0 <= y < src_max_y:
                    # Interpolate without weight
                    x1 = max(math.floor(x), src_min_x)
                    x2 = min(math.ceil(x), src_max_x-1)
                    y1 = max(math.floor(y), src_min_y)
                    y2 = min(math.ceil(y), src_max_y-1)
                    # temp = src[x1, y1, :] + src[x1, y2, :] + src[x2, y1, :] + src[x2, y2, :]
                    # res[target[0], target[1], :] = temp//4
                    res[target[0], target[1], :] = src[x2, y2, :]
    for row in range(dst_min_x, dst_max_x):
        for col in range(dst_min_y, dst_max_y):
            target = np.dot(translation, arr([row, col, 1]))
            temp = dst[row, col, :]
            if res[target[0], target[1], 0] != 0 or res[target[0], target[1], 1] != 0 or res[target[0], target[1], 2] != 0:
                temp = res[target[0], target[1], :] + temp
                temp //= 2
            res[target[0], target[1], :] = temp
    return res


