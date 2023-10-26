import cv2
import numpy as np
from visualization import visualize
from brief import Brief
from fast import FeaturesAcceleratedSegmentTest


if __name__ == "__main__":
    # Load the image
    image1 = cv2.imread('./../image_pairs/image_pairs_01_01.jpg')
    image2 = cv2.imread('./../image_pairs/image_pairs_01_02.jpg')
    # Convert  image to RGB
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    image_cp = np.copy(img_gray)
    image2_cp = np.copy(img2)
    keypoints = FeaturesAcceleratedSegmentTest(img_gray)
    keypoints2 = FeaturesAcceleratedSegmentTest(img2)
    brief = Brief(kernel_size=25, bits=128)
    descriptor_1, location_1 = brief.compute(image=image_cp, keypoints=keypoints)
    descriptor_2, location_2 = brief.compute(image=image2_cp, keypoints=keypoints2)
    pairs = brief.get_threshold_pairs(descriptor_1, descriptor_2, location_1, location_2, 15)
    image1 = cv2.imread('./../image_pairs/image_pairs_01_01.jpg')
    image2 = cv2.imread('./../image_pairs/image_pairs_01_02.jpg')
    visualize(image1, image2, pairs)

