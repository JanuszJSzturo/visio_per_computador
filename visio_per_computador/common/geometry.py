# -*- coding: utf-8 -*-
""" Module containing a set of function to work with descriptors and geometry
"""
import cv2
import numpy as np
from typing import Tuple

ORG_2_HMG = 1
HMG_2_ORG = 2





def alpha_z(b, t, l, v, p=1):
    """
    Function that implements the equation presented in the paper Criminisi et al., 2000, available at:
    https://www.cs.cmu.edu/~ph/869/papers/Criminisi99.pdf
    alpha*Z = -||bxt|| / (l*b)||vxt||

    If parameter p is not passed, it returns the distance between planes scaled to a common factor alpha,
    otherwise it returns it returns either Z or alpha, if alpha or Z have been passed as argument.

    Args:
        b: bottom point of object in homogenous coordinates
        t: top point of object in homogenous coordinates
        l: vanishing line in in homogenous coordinates
        v: vanishing point in z direction
        p: either alpha or Z


    """
    cp1 = np.cross(b, t)
    cp2 = np.cross(v, t)
    dp = np.dot(l, b)
    result = -np.linalg.norm(cp1) / (p * dp * np.linalg.norm(cp2))
    return result


def calculateHeight(img, knownZ, ref_b, ref_t, est_b, est_t, A, B, C, D):
    """

    Args:
        img:
        knownZ:  height of the known object
        ref_b, ref_t: bottom point and top point of the knownZ object
        est_b, est_t: bottom point and top point of the desired height to be calculated
        A,B,C,D: points of the parallelogram in the reference plane set out as follows:
        B---D
        |   |
        A---C
        These points can be rotated but not change positions between them

    Returns: height of the desired object and vanishing points
    """
    #Reference plane points
    A_h = convert(A, img, ORG_2_HMG)
    B_h = convert(B, img, ORG_2_HMG)
    C_h = convert(C, img, ORG_2_HMG)
    D_h = convert(D, img, ORG_2_HMG)

    #Bottom and top points of reference object and the one we want to estimate
    refB_h = convert(ref_b, img, ORG_2_HMG)
    refT_h = convert(ref_t, img, ORG_2_HMG)
    estB_h = convert(est_b, img, ORG_2_HMG)
    estT_h = convert(est_t, img, ORG_2_HMG)

    #Vanishing points and vanishing line
    vp1_h = np.cross(np.cross(A_h, B_h), np.cross(C_h, D_h))
    vp2_h = np.cross(np.cross(A_h, C_h), np.cross(B_h, D_h))
    vl_h = np.cross(vp2_h / vp2_h[2], vp1_h / vp1_h[2])

    #Vertical vanishing point
    vzEst_h = np.cross(estB_h, estT_h)
    vzRef_h = np.cross(refB_h, refT_h)
    vz_h = np.cross(vzEst_h, vzRef_h)

    alpha = alpha_z(refB_h, refT_h, vl_h, vz_h, knownZ)
    height = alpha_z(estB_h, estT_h, vl_h, vz_h, alpha)

    vp1_o = convert(vp1_h, img, HMG_2_ORG)
    vp2_o = convert(vp2_h, img, HMG_2_ORG)
    im = cv2.line(img, vp1_o, vp2_o, (255, 0, 255), 10)

    return height, vp1_o, vp2_o

def my_segment(img, filled, empty, color, thickness, radius):
    """
    Custom function that draws a segment with a filled circle on one end and an empty circle in the other end.

    Args:
        img: the image where to draw the segment
        filled: coordinates of the filled end
        empty: coordinates of the empty end
        color: color of the segment and circles
        thickness: thickness of the segment and circles
        radius:  radius of the circles

    Returns:

    """
    im = cv2.line(img, filled, empty, color, thickness)
    im = cv2.circle(img, filled, radius, color, -1)
    im = cv2.circle(img, empty, radius, color, thickness)
    return im


def convert(coordinates, img: np.ndarray, code: int):
    """ Convert coordinates from image system to homogenous system and vice
    versa.

    The function converts an input coordinate from coordinate system to another.
    To do so we need the shape of the images.

    Args:
        coordinates:
        img:
        code:

    Returns:

    """
    if code != ORG_2_HMG and code != HMG_2_ORG:
        raise TypeError("Unknown operation")

    if code == ORG_2_HMG:
        return __to_homogenous(coordinates, (img.shape[1], img.shape[0]))
    elif code == HMG_2_ORG:
        return __to_original(coordinates, (img.shape[1], img.shape[0]))


def __to_homogenous(coordinates: Tuple[int, int], size: Tuple[int, int]):
    """ Converts coordinates from an image to homogenous system.

    Args:
        coordinates:
        size:

    Returns:

    """
    semi_axis = np.array(size) / 2

    w = ((size[0] + size[1]) / 4)

    homogenous = np.array([coordinates[0] - semi_axis[0], coordinates[1] - semi_axis[1]])
    homogenous = np.append(homogenous, [w])

    return tuple(homogenous)


def __to_original(coordinates: Tuple[int, int], size: Tuple[int, int]):
    semi_axis = np.array(size) / 2

    w = ((size[0] + size[1]) / 4)

    original_x = int(((coordinates[0] / coordinates[-1]) * w) + semi_axis[1])
    original_y = int(((coordinates[1] / coordinates[-1]) * w) + semi_axis[0])

    return original_x, original_y


def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray, lines, pts1, pts2):
    """ We draw a set of lines and points on the images passed as parameter.

    Args:
        img1 (np.ndarray): First image to draw the epipolar images
        img2 (np.ndarray): Second image to draw the epipolar images
        lines (List): A set of epipolar lines
        pts1 (List[Tuple[int,int]]): A list of points in the first image. The
            first part of a matching
        pts2 (List[Tuple[int,int]]): A list of points in the second image. The
            second part of a matching

    Returns:

    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def get_kp_desc(method: str, img: np.ndarray, **kwargs):
    """ Gets the keypoints and the descriptors from an image.

    Calculates the keypoints of an image and its descriptors. To do so we use
    different well-known algorithms. The method used is passed as parameter.
    The results are always a tuple with two elements, the keypoints and its
    descriptors


    Args:
        method (str): See below for more information
        img (np.ndarray): Image to extract the descriptors
        **kwargs: Extra arguments for the methods
    Methods:
        "O" (ORB): Oriented FAST and rotated BRIEF descriptors
        "sift": Scale-Invariant Feature Transform descriptors
        "surf": Speeded Up Robust Features
        "fast_brief": fast (Features from Accelerated Segment Test) keypoints
                      brief( Binary Robust Independent Elementary Features) descriptors
    Returns:
        Tuple with the descriptors and the descriptions
    """
    method_call = None

    if method == "O":
        orb = cv2.ORB_create(**kwargs)
        method_call = orb.detectAndCompute
    elif method == "sift":
        sift = cv2.xfeatures2d.SIFT_create(**kwargs)
        method_call = sift.detectAndCompute
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create(**kwargs)
        method_call = surf.detectAndCompute

    if method == "fast_brief":
        fast = cv2.FastFeatureDetector_create(**kwargs)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(**kwargs)
        kp = fast.detect(img, None)
        kp, descs = brief.compute(img, kp)
    else:
        kp, descs = method_call(img, mask=None, **kwargs)

    return kp, descs


def match_descriptors(method: str, desc1, desc2, **kwargs):
    """ Search matches between two sets of descriptors

    Search matches between two set of descriptors. Multiple methods are
    available. The usefulness of the methods depend on the format of the
    descriptors.

    Args:
        method (str): See options below
        desc1: List of descriptors
        desc2: List of descriptors
    Methods:
        "BF_H": Bruteforce matcher with Hamming.
        "BF_k": Bruteforce without Hamming taking k nearest neighbours.
        "BF": Bruteforce without Hamming.
        "BF_H_k": Bruteforce matcher with Hamming taking k nearest neighbours.
    Returns:
        List of matches
    """
    matches = None
    if method == "BF_H":
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(desc1, desc2, **kwargs)
    elif method == "BF_k":
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc1, desc2, **kwargs)
    elif method == "BF":
        matcher = cv2.BFMatcher()
        matches = matcher.match(desc1, desc2, **kwargs)
    elif method == "BF_H_k":
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(desc1, desc2, **kwargs)

    return matches


def filter_matches(method: str, matches, min_distance: int = None,
                   proportion: float = None):
    """ Filter matches by multiple conditions.

    Args:
        method (str): See below list of methods available
        matches: List of matches
        min_distance (int): Minimum distance to accept a match (DIST)
        proportion (float): Proportion between the second and first match
                            distances (KNN)

    Methods:
        "DIST": Select the matches with a distance higher than a minimum
        "KNN": Select the matches with a small distance on the first match and
                with a high value on the second. Defined by proportion

    Returns:
        List of filtered matches
    """
    if method == "DIST":
        matches = list(filter(lambda m: m.distance < min_distance, matches))
    elif method == "KNN":
        matches = list(filter(lambda m: m[0].distance < m[1].distance * proportion, matches))

    return matches
