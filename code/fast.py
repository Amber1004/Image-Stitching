from skimage import img_as_float


def FeaturesAcceleratedSegmentTest(gray_img):
    t = 0.05
    offsets = [(0, 3), (-1, 3), (1, 3), (2, 2), (3, 0), (3, 1), (3, -1), (2, -2), 
               (0, -3), (1, -3), (-1, -3), (-2, -2), (-3, 0), (-3, 1), (-3, -1), (-2, 2)]
    # # Load the image
    # image1 = cv2.imread('./image_pairs/image_pairs_02_01.png')
    # # Convert  image to RGB
    # img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # # Convert image to gray scale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    image = img_as_float(gray_img)

    keyPoints = []
    tmp = []
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            count=0
            for x,y in [offsets[0], offsets[4], offsets[8], offsets[12]]:
                x_off = i+x
                y_off = j+y
                if (x_off >= 0 and x_off <= image.shape[0]-1) and (y_off >= 0 and y_off <= image.shape[1]-1):
                    if abs(image[i][j] - image[i+x][j+y]) >= t:
                        count+=1
                #print(f"{(i, j)}{image[i][j]} {(x,  y)} {image[i+x][j+y]} {count}")
            if count>=3:
                count = 0
                for x,v in offsets:
                    x_off = i+x
                    y_off = j+y
                    if (x_off >= 0 and x_off <= image.shape[0]-1) and (y_off >= 0 and y_off <= image.shape[1]-1):
                        if abs(image[i][j] - image[i+x][j+y]) >= t:
                            count+=1
                if count>=12:
                    # keyPoints.append(cv2.KeyPoint(j, i, 0.1))
                    tmp.append((i, j))
                #if i>400:
                #    print(f"{(i, j)}{image[i][j]} {count}")
            #print('\n')
    #print(tmp)
    return tmp

# keyPoints = tuple(FeaturesAcceleratedSegmentTest())


# # Load the image
# image1 = cv2.imread('./image_pairs/image_pairs_02_01.png')
# # Convert  image to RGB
# img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image_kp = np.copy(img)
# cv2.drawKeypoints(img, keyPoints, image_kp, color = (0, 255, 0))

# fx, plots = plt.subplots(1, 2, figsize=(20,10))

# plots[0].imshow(img)

# plots[1].imshow(image_kp)

# plt.show()

