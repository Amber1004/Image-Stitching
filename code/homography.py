import numpy as np
import cv2
import math

# Get random pairs of matching points
def random_pairs(src_pts, des_pts, n=4):

    src_random = []
    des_random = []

    index = np.random.choice(len(src_pts), n, replace=False)
    index_list = index.tolist()
    for i in range (len(index_list)):
        pts_index = index_list[i]
        src_random.append(tuple(src_pts[pts_index]))
        des_random.append(tuple(des_pts[pts_index]))
    
    return src_random, des_random

# Calculate homography
def calc_homography(src, des):
    A = []
    b = []
    for src_pt, dst_pt in zip(src, des):
        x, y = src_pt
        u, v = dst_pt
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v])
        b.append([u])
        b.append([v])
    A = np.array(A)
    b = np.array(b)

    # Least Square Solution
    h = np.linalg.lstsq(A, b, rcond=None)[0]

    H = []
    for i in h:
        H.append(i[0])
    H.append(1)
    H = np.array(H).reshape(3,3)
    return H

# Apply homography on the source point to get the estimated destination point
def apply_homography(src, H):
    src_arr = np.array(src)
    src_arr = src_arr.reshape(-1, src_arr.shape[-1])
    src_3d = np.ones((len(src_arr)))
    src_homogenous = np.column_stack((src_arr,src_3d))
    homography_point = np.dot(src_homogenous,H.T)
    src_homo_pts = homography_point[:,:2]/homography_point[:,2,np.newaxis]
    return src_homo_pts
    
# Find the Euclidean distance between the estimated des pt and the matching des pt
def error_distance(des, homo_des):
    # Calculate Euclidean distance between the matched destination point and the homography calculated destination point
    error = []
    for des_pt, homo_pt in zip(des, homo_des):
        x1, y1 = des_pt
        x2, y2 = homo_pt
        err = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        error.append(err)
    
    return error
    
def ransac(pairs, number_iteration=2000, threshold=3):
    """
        Homography calculation
        :param pairs: matching points between image 1 and image 2
        :param number_iteration: number of iteration for RANSAC algorithm
        :param threshold: threshold for counting the inliers 
        :return: 3x3 homography matrix
    """

    src_pts = []
    des_pts = []
    for p in pairs:
        src_pts.append(p[0])
        des_pts.append(p[1])
        
    most_inliers_count = 0
    best_inliers = None
    best_H = None

    for i in range (number_iteration):
        # Get 4 random pairs of matching points to calculate the homography
        src_random, des_random = random_pairs(src_pts, des_pts, 4)
        # Calculate the homography
        H = calc_homography(src_random, des_random)
        # Transform all the source point using the homography calculated
        homography_pts = apply_homography(src_pts, H)
        # Calculate the error between the points from homography calculation and the destination points
        error = error_distance(des_pts, homography_pts)

        # Count the inliers (errors less than threshold)
        inliers = []
        for i in range (len(error)):
            if error[i] < threshold:
                inliers.append(i)
        inliers_count = len(inliers)

        # Keep the largest set of inliers
        if inliers_count > most_inliers_count:
            best_H = H
            most_inliers_count = inliers_count
            best_inliers = inliers
            
# Recompute the homography using all the inliers in the best model
    src_inliers = []
    des_inliers = []
    for i in range(best_inliers):
        src_inliers.append(src_pts[i])
        des_inliers.append(des_pts[i])

    best_H = calc_homography(src_inliers,des_inliers)

    return best_H