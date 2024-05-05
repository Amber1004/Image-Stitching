import cv2
import numpy as np
from visualization import visualize, show_img
from brief import Brief
from fast import FeaturesAcceleratedSegmentTest
from orb import orb
from homography import ransac
from image_warpping import image_warping


im_pairs = {
    0: ['image_pairs_01_01.jpg', 'image_pairs_01_02.jpg'],
    1: ['image_pairs_02_01.png', 'image_pairs_02_02.png'],
    2: ['image_pairs_03_01.jpg', 'image_pairs_03_02.jpg'],
    3: ['image_pairs_04_01.jpg', 'image_pairs_04_02.jpg'],
}

if __name__ == "__main__":
    # Please change your_path to the correct directory
    your_path = "./../image_pairs/"
    # this is the image pair you want to run
    pair_id = 1
    # Load the image
    image1 = cv2.imread(your_path+im_pairs[pair_id][0])
    image2 = cv2.imread(your_path+im_pairs[pair_id][1])
    print(your_path+im_pairs[pair_id][0], your_path+im_pairs[pair_id][1])
    # Convert  image to RGB
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # # #ORB
    pairs = orb(img_gray, img2_gray, brief_kernel_size=30, bit_length=128, threshold=6)
    # Homography
    h, src_in, dst_in = ransac(pairs, number_iteration=2000, threshold=2)
    wrapped_img = image_warping(img2, img, h)
    show_img(wrapped_img)
    in_pairs = [(i, j) for i, j in zip(src_in, dst_in)]
    visualize(image1, image2, in_pairs)
