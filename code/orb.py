from fast import FeaturesAcceleratedSegmentTest
from brief import Brief
from visualization import visualize
import numpy as np


def orb(gray_img1, gray_img2, brief_kernel_size, bit_length, threshold):
    """
    (Oriented FAST and Rotated BRIEF)
    :param gray_img1: gray scale image
    :param gray_img2: gray scale image
    :param brief_kernel_size: the kernel size used in BRIEF
    :param bit_length: 128 or 256 bits for BRIEF descriptor
    :param threshold: threshold for hamming distance
    :return: a list of point pairs list(tuple(tuple(x,y), tuple(x,y)), ...)
    """
    keypoints = FeaturesAcceleratedSegmentTest(np.copy(gray_img1))
    keypoints2 = FeaturesAcceleratedSegmentTest(np.copy(gray_img2))
    brief = Brief(kernel_size=brief_kernel_size, bits=bit_length)
    descriptor_1, location_1 = brief.compute(image=np.copy(gray_img1), keypoints=keypoints)
    descriptor_2, location_2 = brief.compute(image=np.copy(gray_img2), keypoints=keypoints2)
    pairs = brief.get_threshold_pairs(descriptor_1, descriptor_2, location_1, location_2, threshold)
    visualize(gray_img1, gray_img2, pairs)
    return pairs


