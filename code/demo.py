import cv2
import numpy as np
import matplotlib.pyplot as plt
# from brief import brief


if __name__ == '__main__':
    img = cv2.imread("./../image_pairs/image_pairs_01_01.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    test_image = cv2.pyrDown(img)
    test_image = cv2.pyrDown(test_image)
    num_rows, num_cols = test_image.shape[:2]
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)

    # Display traning image and testing image
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plots[0].set_title("Training Image")
    plots[0].imshow(img)

    plots[1].set_title("Testing Image")
    plots[1].imshow(test_image)

    # ## Detect keypoints and Create Descriptor

    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    train_keypoints = fast.detect(img_gray, None)
    test_keypoints = fast.detect(test_gray, None)
    # pts = np.float([key_point.pt for key_point in train_keypoints]).reshape(-1, 1, 2)
    train_keypoints_arr = np.array([key_point.pt for key_point in train_keypoints]) # keypoint to array

    train_keypoints, train_descriptor = brief.compute(img_gray, train_keypoints)
    test_keypoints, test_descriptor = brief.compute(test_gray, test_keypoints)
    print(train_descriptor[0])
    # train_keypoints, train_descriptor = brief(img_gray, train_keypoints)
    # test_keypoints, test_descriptor = brief(test_gray, test_keypoints)

    keypoints_without_size = np.copy(img)
    keypoints_with_size = np.copy(img)

    cv2.drawKeypoints(img, train_keypoints, keypoints_without_size, color=(0, 255, 0))

    cv2.drawKeypoints(img, train_keypoints, keypoints_with_size,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    # Print the number of keypoints detected in the training image
    print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

    # Print the number of keypoints detected in the query image
    print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

    # ## Matching Keypoints

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Perform the matching between the BRIEF descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(img, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags=2)

    # Display the best matching points
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

    # Print total number of matching points between the training and query images
    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

