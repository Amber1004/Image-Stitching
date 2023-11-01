import cv2
import numpy as np
from visualization import visualize, show_img
from brief import Brief
from fast import FeaturesAcceleratedSegmentTest
from orb import orb
from homography import ransac
from image_warpping import image_warping

if __name__ == "__main__":
    # Load the image
    image1 = cv2.imread('./image_pairs/image_pairs_03_01.jpg')
    image2 = cv2.imread('./image_pairs/image_pairs_03_02.jpg')
    # Convert  image to RGB
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # # #ORB
    pairs = orb(img_gray, img2_gray, brief_kernel_size=25, bit_length=128, threshold=10)
    # Homography
    h = ransac(pairs, number_iteration=2000, threshold=3)
    print(h)
    # h = np.array([[0.7655593917059154, -0.398028467888191, 27.680515613838626], [0.056345642328405145, 0.4733382440498921, 196.23207266953582], [-0.00018580192975118227, -0.0012089784956920817, 1.0]])
    wrapped_img = image_warping(img2, img, h)
    show_img(wrapped_img)
    # h = [[0.7655593917059154, -0.398028467888191, 27.680515613838626], [5.63456423e-02, 4.73338244e-01, 1.96232073e+02], [-1.85801930e-04, -1.20897850e-03, 1.00000000e+00]]
    # print(h)
