from typing import List, Tuple, Dict, Optional
from logging import Logger
from pathlib import Path
import os
# from unittest import signals
import numpy as np
from skimage.transform import resize
from tifffile import imread
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import skew

def getTraceByPlace(data: np.ndarray, segments:np.ndarray,frameRange:Optional[tuple[int, int]], baselineImage:Optional[np.ndarray]=None, byFrameCentered:Optional[bool]=True)->np.ndarray:
    """
    Docstring for getTraceByPlace
    
    :param data: in shape [number of frames, width, height]
    :type data: np.ndarray
    :param segments: in shape [number of segments, width, height]
    :type segments: np.ndarray
    :param frameRange: Description
    :type frameRange: Optional[tuple[int, int]]
    :param baselineImage: in shape [width, height]
    :type baselineImage: Optional[np.ndarray]
    :param byFrameCentered: Description
    :type byFrameCentered: Optional[bool]
    :return: Description
    :rtype: ndarray
    """