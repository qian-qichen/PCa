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
from scipy.ndimage import gaussian_filter, label, center_of_mass
from numpy.lib import scimath

ICA_SK = True
FASTICA_ALGORITHM = 'parallel'  # 'parallel' or 'deflation'
FASTICA_FUN = 'logcosh'  # 'logcosh', 'exp', 'cube', 'gauss'

def ica_manual(data:np.ndarray, num_IC:Optional[int]=None, ica_guess:Optional[np.ndarray]=None, termtol:Optional[float]=1e-6, maxrounds:Optional[int]=100):
    """
    docstring for ica function

    :param data: in shape [number of PCs, number of samples]
    :type data: np.ndarray
    :param num_IC:  < number of PCs
    :type num_IC: int
    :param ica_guess: in shape [number of PCs, number of ICs]
    :type ica_guess: np.ndarray
    :param termtol: 
    :type termtol: float
    :param maxrounds: 
    :type maxrounds: int
    """
    num_PC = data.shape[0]
    number_sample = data.shape[1]
    if num_IC is None:
        num_IC = num_PC
    if ica_guess is None or ica_guess.shape != (num_PC, num_IC):
        ica_guess = np.random.rand(num_PC, num_IC)
        u, s, vt = np.linalg.svd(ica_guess, full_matrices=False)
        ica_guess = u @ vt
        # ica_guess = ica_guess @ np.real(scimath.sqrt(np.linalg.inv(ica_guess.T @ ica_guess)))

    iterCount = 0
    old = ica_guess.copy()
    loss = 1
    while iterCount < maxrounds and loss > termtol:
        iterCount += 1
        new = (data @ (data.T @ old)**2) / number_sample
        u, s, vt = np.linalg.svd(new, full_matrices=False)
        new = u @ vt

        loss = 1 - np.min(np.abs(np.diag(new.T @ old)))
        old = new.copy()
    return new, iterCount, loss


        

def cellsortICA(mixedSig:np.ndarray, mixedFilters:np.ndarray, singlar_values:np.ndarray, mu:float, num_IC:Optional[int]=None, termtol:Optional[float]=1e-6, maxrounds:Optional[int]=100,PC_use:Optional[List[int]]=None, ica_A_guess:Optional[np.ndarray]=None,logger: Optional[Logger]=None)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    docstring for CellsortICA function
    :param mixedsig: in shape [number of PCs, number of time points]
    :type mixedsig: np.ndarray
    :param mixedfilters: in shape [number of PCs, width, height]
    :type mixedfilters: np.ndarray
    :param singlar_values: [number of PCs]
    :type singlar_values: np.ndarray
    :param mu:  in range [0,1]
    :type mu: float
    :param numIC:  < len(PC_use) < number of PCs
    :type numIC: int
    :param termtol: docstring for termtol
    :type termtol: float
    :param maxrounds: docstring for maxrounds
    :type maxrounds: int
    :param PC_use: 
    :type PC_use: List[int]
    :param ica_A_guess: docstring for ica_A_guess
    :type ica_A_guess: np.ndarray
    :param logger: Optional logger for logging messages
    :type logger: Logger
    :return:
        - ica_sig: in shape [number of ICs, number of time points]
        - ica_filters: in shape [number of ICs, number of pixels]
        - ica_A: in shape [number of PCs, number of ICs]
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    ## TODO check inputs here
    ## TODO create a shell logger if not provided
    ## select PCs
    if PC_use is not None:
        mixedSig = mixedSig[PC_use, :]
        mixedFilters = mixedFilters[PC_use, :]
        singlar_values = singlar_values[PC_use]
    num_PC, num_frame = mixedSig.shape
    num_pix = mixedFilters.shape[1]

    ## cetering
    meanSig = mixedSig.mean(axis=1, keepdims=True)
    centeredSig = mixedSig - meanSig
    del mixedSig
    ## concatenate
    if mu == 0:
        concatSig = mixedFilters
    elif mu == 1:
        concatSig = centeredSig
    else:
        concatSig = np.concatenate([(1-mu)*mixedFilters, mu*centeredSig], axis=1) # (num_PC, num_pix + num_frame)
        concatSig = concatSig / np.sqrt(2*(1-2*mu+mu**2)) # keep covariance as unit if both parts have unit covariance
    corr = concatSig @ concatSig.T 
    Uerr = np.sum(np.abs(corr - np.eye(corr.shape[0])))
    logger.info(f"Initial concatenated signal covariance deviation from identity: {Uerr:.4e}")
    ## ICA
    if ICA_SK:
        ica = FastICA(algorithm=FASTICA_ALGORITHM,whiten=False,fun=FASTICA_FUN,max_iter=maxrounds,tol=termtol, w_init=ica_A_guess)
        ica.fit(concatSig.T)
        ica_mat = ica.components_ # [number of IC, number of PC], number of IC == number of PC, S = ica_mar @ X
        ica_signal = ica.transform(concatSig.T).T # (num_PC, num_pix + num_frame)
        # ica_signal = ica_mat @ concatSig 

        if num_IC is not None:
            raise NotImplementedError("sklearn ICA method with num_IC is not implemented yet")
    else:
        ica_mat, iterCount, loss = ica_manual(concatSig, num_IC=num_IC, ica_guess=ica_A_guess, termtol=termtol, maxrounds=maxrounds)
        corr = ica_mat @ ica_mat.T


        if loss > termtol:
            logger.warning(f"ICA did not converge within {iterCount} iterations, final loss: {loss}")
        logger.info(f"ICA done in {iterCount} iterations, final loss: {loss}")
        ica_signal = ica_mat.T @ concatSig  # [number of ICs, ...]
    skew_values = np.abs(skew(ica_signal, axis=1))
    order = np.argsort(-skew_values)
    # print(order)
    ica_signal = ica_signal[order, :]
    ica_mat = ica_mat[:,order]
    skew_values = skew_values[order]

    ica_sig = ica_mat.T @ centeredSig  # [number of ICs, number of time points]
    # ica_failter = ica_mat.T @ (mixedFilters * singlar_values.reshape(-1,1))  # [number of ICs, number of pixels]
    ica_failter = ica_mat.T @ mixedFilters        

    return ica_sig, ica_failter, ica_mat

def cellsortSegment(ica_filters:np.ndarray,smwidth:Optional[float]=None,thresh:Optional[float]=2, areaLimits:Tuple[int, int]|int=200)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Docstring for cellsortSegment
    
    :param ica_filters: in shape [number of ICs, width, height]
    :type ica_filters: np.ndarray
    :param smwidth: standard deviation of Gaussian smoothing kernel
    :type smwidth: Optional[float]
    :param thresh: 
    :type thresh: Optional[float]
    :param areaLimits: (min area, max area) or min area, means max in pixels, for segmentation mask
    :type areaLimits: Tuple[int, int] | int
    :return: 
        - seg_filters: in shape [number of segments, width, height]
        
        - seg_areas: in shape [number of segments]
        
        - seg_centroids: in shape [number of segments, 2] , (row, column)
    :rtype: Tuple[ndarray, ndarray, ndarray]
    """
    if isinstance(areaLimits, int):
        min_area = int(areaLimits)
        max_area = None
    else:
        min_area = int(areaLimits[0])
        max_area = int(areaLimits[1]) if len(areaLimits) > 1 else None
    numIC, w, h = ica_filters.shape
    ## gobal normalization
    global_std = np.abs(np.std(ica_filters))
    if global_std == 0:
        global_std = 1.0
    ica_filtersorig = ica_filters / global_std
    global_mean = np.mean(ica_filters)
    ica_norm = (ica_filters - global_mean) / global_std
    
    ## thresholding with optional local normalization
    masks = np.zeros_like(ica_filters, dtype=bool)
    if smwidth is not None:
        for j in range(numIC):
            ic = ica_norm[j, :, :].copy()
            # local normalization per IC as in MATLAB loop
            ic_mean = np.mean(ic)
            ic_std = np.abs(np.std(ic))
            if ic_std == 0:
                ic_std = 1.0
            ic = (ic - ic_mean) / ic_std
            ic_s = gaussian_filter(ic, sigma=smwidth, mode='nearest')
            masks[j, :, :] = ic_s > thresh
    else:
        masks = ic > thresh
    ## segmentation
    # seg_filters = np.zeros_like(ica_filters,dtype=np.bool)
    # segment_labels = np.zeros((numIC,), dtype=np.int32) - 1
    # seg_centroids = np.zeros((numIC,2), dtype=np.float32) - 1
    segments,segment_labels,seg_centroids = [], [], []
    num_accepted = 0
    for i, IC in enumerate(masks):
        labelMaps, num_labels = label(IC)
        for l in range(1, num_labels+1):
            seg_mask = labelMaps == l
            area = np.sum(seg_mask)
            if area < min_area:
                continue
            if (max_area is not None) and (area > max_area):
                continue
            seg_filter = seg_mask * ica_filtersorig[i]
            segments.append(seg_filter)
            segment_labels.append(num_accepted)
            cy, cx = center_of_mass(seg_mask)
            seg_centroids.append((cy, cx))
            num_accepted += 1
    
    segments = np.stack(segments,axis=0) ## TODO turn to spare mat?
    segment_labels = np.array(segment_labels)
    seg_centroids = np.array(seg_centroids)



    return segments, segment_labels, seg_centroids


def cellsortSegmentPlot(seg_filters: np.ndarray, seg_centroids: np.ndarray, seg_labels: Optional[np.ndarray] = None, background: Optional[np.ndarray] = None, cmap: str = 'gray') -> Figure:
    """
    Simple plotting helper to show segmentation results.

    - overlays contours of each segment on `background` if provided
    - otherwise shows a montage of segment filters

    :param seg_filters: (nSeg, W, H) segment images
    :param seg_centroids: (nSeg, 2) centroids (row, col)
    :param seg_labels: optional labels for segments
    :param background: optional (W, H) background image
    :param cmap: matplotlib colormap name
    :return: matplotlib Figure
    """
    nSeg = seg_filters.shape[0]
    fig = plt.figure(figsize=(8, 6))
    if background is not None:
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(background, cmap=cmap)
        for i in range(nSeg):
            img = gaussian_filter(seg_filters[i], sigma=1.0)
            level = img.mean() + 2.5 * img.std()
            ax.contour(img, levels=[level], colors=[plt.cm.tab10(i % 10)])
            cy, cx = seg_centroids[i]
            ax.text(cx, cy, str(i + 1), color='y', ha='center', va='center')
        ax.axis('off')
    else:
        # montage: arrange in grid
        cols = int(np.ceil(np.sqrt(nSeg)))
        rows = int(np.ceil(nSeg / cols))
        for i in range(nSeg):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(seg_filters[i], cmap='hot')
            ax.set_title(str(i + 1))
            ax.axis('off')
    fig.tight_layout()
    return fig


def ICAshow(ica_filter, backgroundImage,figSize,scale=4,cmap='jet')->Figure:
    fig, ax = plt.subplots(figsize=figSize)
    ax.imshow(backgroundImage, cmap=cmap)
    numIC = ica_filter.shape[0]
    crange = [np.min(ica_filter), np.max(ica_filter)]
    contourlevel = crange[1] - np.diff(crange)*0.8
    for i , IC in enumerate(ica_filter):
        IC = gaussian_filter(IC, sigma=1.0,mode='nearest')
        show_Level = np.mean(IC) + scale * np.std(IC)
        # show_Level = np.percentile(IC, scale)
        # print(show_Level)
        peakPoint = np.unravel_index(np.argmax(IC), IC.shape)
        ax.contour(IC, levels=[show_Level], colors='r')
        ax.text(peakPoint[1], peakPoint[0], str(i + 1), color='y', ha='center', va='center')


    return fig












