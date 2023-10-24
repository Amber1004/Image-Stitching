import numpy as np


class Brief:

    def __init__(self, kernel_size=21, bits=128):
        """
        This Brief class used gaussian distribution to select point pairs.
        :param kernel_size: The rectangular box used for every keypoint, better to be odd number
        :param bits: 128 or 256 bits 
        """
        self.kernel = self.__gaussian_kernel(kernel_size)
        self.kernel_size = kernel_size
        self.bits = bits
        self.pt_pairs = []
        flatten_kernel = self.kernel.flatten()
        pos = np.arrange(0, len(flatten_kernel))
        for i in range(bits):
            random_a = np.random.choice(pos, p=flatten_kernel)
            random_b = np.random.choice(pos, p=flatten_kernel)
            self.pt_pairs.append((random_a, random_b))

    @staticmethod
    def __gaussian_kernel(kernel_size, std=1, mean=0):
        # Initializing value of x,y as grid of kernel size
        # in the range of kernel size

        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                           np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x ** 2 + y ** 2)

        # lower normal part of gaussian
        normal = 1 / (2, 0 * np.pi * std ** 2)

        # Calculating Gaussian filter
        gauss = np.exp(-((dst - mean) ** 2 / (2.0 * std ** 2))) * normal
        return gauss

    def __describe(self, image, x, y):
        rectangle = image[x-self.kernel_size/2:x+self.kernel_size/2, y-self.kernel_size/2:y+self.kernel_size/2]
        flatten_rectangle = rectangle.flatten()
        bit_str = ""
        for pair in self.pt_pairs:
            bit_str += "1" if flatten_rectangle[pair[0]] < flatten_rectangle[pair[1]] else "0"
        return int(bit_str, base=2)

    def compute(self, image, keypoints):
        """
        BRIEF detector converts the given keypoints into 128 or 256 bit encoding
        :param image: a gray scale image 2 dim array (HxW)
        :param keypoints: an array of keypoint positions
        :return: descriptor, keypoint position
        """
        descriptor = []
        for keypoint in keypoints:
            descriptor.append(self.__describe(image, keypoint[0], keypoint[1]))
        return descriptor, keypoints



