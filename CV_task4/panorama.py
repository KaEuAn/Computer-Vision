import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp, rescale
from skimage.filters import gaussian
from numpy.linalg import inv
from math import sqrt
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=350):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    brightness = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    detector_extractor = ORB(n_keypoints=n_keypoints)
    detector_extractor.detect_and_extract(brightness)
    return detector_extractor.keypoints, detector_extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))

    Cx = np.mean(points[:, 0])
    Cy = np.mean(points[:, 1])
    distance = np.mean(np.sqrt((points[:, 0] - Cx) ** 2 + (points[:, 1] - Cy) ** 2))
    N = sqrt(2) / distance
    matrix[0, 0] = N; matrix[1, 1] = N; matrix[2, 2] = 1
    matrix[0, 2] = -N * Cx; matrix[1, 2] = -N * Cy
    return matrix, (matrix @ pointsh).T
    #everything is ok, tested


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)
    n = src_keypoints.shape[0]


    A = np.zeros((n * 2, 9))
    for i in range(0, n * 2, 2):
        j = i // 2
        A[i] = [-src[j, 0], -src[j, 1], -1, 0, 0, 0, dest[j, 0] * src[j, 0], dest[j, 0] * src[j, 1], dest[j, 0]]
        A[i + 1] = [0, 0, 0, -src[j, 0], -src[j, 1], -1, dest[j, 1] * src[j, 0], dest[j, 1] * src[j, 1], dest[j, 1]]
    u, s, v = np.linalg.svd(A)

    h = v[-1]

    H = np.resize(h, (3, 3))

    H = np.linalg.inv(dest_matrix) @ H @ src_matrix
    return H


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=150,
                     residual_threshold=2.0, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    matches = match_descriptors(src_descriptors, dest_descriptors)
    inliers = []
    for i in range(max_trials):
        choice = matches[np.random.choice(matches.shape[0], 4)]
        H = find_homography(src_keypoints[choice[:, 0]], dest_keypoints[choice[:, 1]])
        new_inliers = []
        for j in range(matches.shape[0]):
            if norm(dest_keypoints[matches[j, 1]] - ProjectiveTransform(H)(src_keypoints[matches[j, 0]])) \
                    < residual_threshold:
                new_inliers.append(j)
        if len(new_inliers) > len(inliers):
            inliers = new_inliers

    H = find_homography(src_keypoints[matches[inliers, 0]], dest_keypoints[matches[inliers, 1]])
    if not return_matches:
        return ProjectiveTransform(H)
    return ProjectiveTransform(H), matches[inliers]


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + ProjectiveTransform(np.linalg.inv(forward_transforms[i - 1].params))
    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i]

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(tuple(corners))
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
            y, x : len of image
        """
    mii, maa = get_min_max_coords(get_corners(image_collection, simple_center_warps))

    center_warps = [None] * len(simple_center_warps)
    shift = np.zeros((3, 3))
    shift[0, 0], shift[1, 1], shift[2, 2] = 1, 1, 1
    shift[0, 2], shift[1, 2] = -mii[1], -mii[0]
    shiftTransform = ProjectiveTransform(shift)
    for i in range(len(simple_center_warps)):
        center_warps[i] = simple_center_warps[i] + shiftTransform
    return center_warps, (int(maa[1] - mii[1]), int(maa[0] - mii[0]))


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    rotated_transform = rotate_transform_matrix(transform)
    transform_matrix = np.linalg.inv(rotated_transform.params)
    ans = warp(image, transform_matrix, output_shape=output_shape)
    return ans, np.any(ans, axis=-1)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for image, fc_warp in zip(image_collection, final_center_warps):
        new_warp, new_mask = warp_image(image, fc_warp, output_shape)
        working_mask = np.logical_not(result_mask)
        working_mask &= new_mask
        working_mask = np.resize(np.repeat(working_mask, 3), output_shape + (3,))
        result[working_mask] = new_warp[working_mask]
        result_mask |= new_mask

    return result


def get_gaussian_pyramid(image, n_layers=4, sigma=0.5):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    layers = [None] * n_layers
    layers[0] = image
    for i in range(1, n_layers):
        layers[i] = gaussian_filter(layers[i - 1][:, :], sigma=sigma)
    return layers


def get_laplacian_pyramid(image, n_layers=4, sigma=0.5):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    layers = get_gaussian_pyramid(image, n_layers=n_layers, sigma=sigma)
    new_layers = [None] * n_layers
    for i in range(n_layers - 1):
        new_layers[i] = layers[i] - layers[i + 1]
    new_layers[-1] = layers[-1]
    return tuple(layers)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=0.5, merge_sigma=0.5):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for image, fc_warp in zip(image_collection, final_center_warps):
        new_warp, new_mask = warp_image(image, fc_warp, output_shape)
        working_mask = np.logical_not(result_mask)
        working_mask &= new_mask
        working_mask_pyramid = get_gaussian_pyramid(working_mask, n_layers=n_layers, sigma=merge_sigma)
        la = get_laplacian_pyramid(result, n_layers=n_layers, sigma=image_sigma)
        lb = get_laplacian_pyramid(new_warp, n_layers=n_layers, sigma=image_sigma)
        l_ans = np.copy(la)
        for i in range(n_layers):
            www = np.resize(np.repeat(working_mask_pyramid[i], 3), output_shape + (3,))
            l_ans[i] = la[i] * www + (1 - www) * lb[i]
            result_mask |= www[:, :, 0]
        result = merge_laplacian_pyramid(l_ans)

    return result
