from typing import List, Tuple, Dict, Optional
from logging import Logger
from pathlib import Path
import os
# from unittest import signals
import numpy as np
from skimage.transform import resize
from tifffile import imread
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
ENABLE_onTime_daul_PCA: bool = False
def dualPCAtransfer(X,PCs, single_values):
    scales = (1.0 / single_values).reshape(1,-1)
    anotherPCs = X @ (PCs.T * scales)
    
    return anotherPCs.T
      
def cellsortPCA(data:Path|str|np.ndarray,numPC:int, frameRange:Optional[Tuple[int,int]]=None, sizeShrink:Optional[float]=None, timeShrink:Optional[float]=None,outputdir:Optional[Path|str]=None,delFrame:Optional[List[int]]=None, logger:Logger=None)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ## TODO more detailed docstring
    """
    Docstring for cellsortPCA
    note: DFoF while not raise error, but still do NOT support data with negative values
    :param data: data input, if a file path is given, the file should be a tif file, or if a numpy array is given, it should be a 3D array with shape (time, height, width)
    :type data: Path | str | np.ndarray
    :param numPC: Description
    :type numPC: int
    :param frameRange: Description
    :type frameRange: Optional[Tuple[int, int]]
    :param sizeShrink: Description
    :type sizeShrink: Optional[float]
    :param timeShrink: Description
    :type timeShrink: Optional[float]
    :param outputdir: Description
    :type outputdir: Optional[Path | str]
    :param delFrame: Description
    :type delFrame: Optional[List[int]]
    :return: 
        - mixedsig: shape (numPC, frames after processing)
        - mixedfilters: shape (numPC, height, width)
        - eigenvalues: shape (numPC,)
        - covtrace: trace of covariance matrix
        - temporal_mean: shape (1, height*width)
        - spacial_mean: shape (frames after processing, 1), after deltaF/F0 normalization

    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    ## TODO add input check here
    ## TODO create a shell logger if not provided
    ## data load
    if isinstance(data, (Path, str)):
        raise NotImplementedError("File path input not implemented yet")
    elif  isinstance(data, np.ndarray):
        if delFrame is not None:
            data = np.delete(data, delFrame, axis=0)
        if frameRange is not None:
            if delFrame is not None:
                before = 0
                middle = 0
                after = 0
                for index in delFrame:
                    if index < frameRange[0]:
                        before += 1
                    elif frameRange[0] <= index < frameRange[1]:
                        middle += 1
                    else:
                        after += 1
                frameRange = (frameRange[0]-before, frameRange[1]-before-middle)
                assert frameRange[0] >= 0, f"frameRange after deleting frames is invalid, range is {frameRange}"
            data = data[frameRange[0]:frameRange[1], :, :]
        if sizeShrink is not None:
            data = resize(data, (int(data.shape[0]), int(data.shape[1]*sizeShrink), int(data.shape[2]*sizeShrink)), order=1)
        if timeShrink is not None:
            data = data[::int(1/timeShrink), :, :]
    else:
        raise ValueError("data must be a file path or a numpy array")
    T, d1, d2 = data.shape
    pixelNum = d1 * d2
    data_2d = data.reshape((T, pixelNum))
    ## DFoF normalization, positions with zero mean will not change
    temporal_mean = np.mean(data_2d, axis=0, keepdims=True)
    data_mean_0 = temporal_mean == 0
    data_mean_dev = temporal_mean.copy()
    data_mean_dev[data_mean_0] = 1
    data_normalized = (data_2d - temporal_mean) / data_mean_dev # (T, pixelNum)
    spacial_mean = np.mean(data_normalized, axis=1, keepdims=True)

    pca = PCA(n_components=numPC,copy=True, svd_solver='full') ## TODO inefficient PCA may slow real data processing
    ## TODO matlab implementation consider negative eigenvalues, maybe need to check sklearn PCA behavior
    if T < pixelNum and ENABLE_onTime_daul_PCA:
        print("PCA on time")
        logger.info("Performing PCA on data with less time points than pixels")
        logger.info("apply temporal covariance matrix")
        pca.fit(data_normalized.T)
        PCs = pca.components_ # (numPC,T)
        mixedsig = PCs # (numPC, T)
        eigenvalues = pca.explained_variance_ # (numPC,)
        single_values = pca.singular_values_  # (numPC,)
        mixedfilters = dualPCAtransfer(data_normalized.T, PCs, single_values) # (numPC, pixelNum)
        # mixedfilters = mixedfilters.reshape((numPC, d1, d2)) # (numPC, d1, d2)
        covtrace = np.trace(pca.get_covariance()) / pixelNum
    else:
        logger.info("Performing PCA on data with less pixels than time points")
        logger.info("apply spatial covariance matrix")
        pca.fit(data_normalized)
        PCs = pca.components_ # (numPC,pixelNum)
        eigenvalues = pca.explained_variance_ # (numPC,)
        single_values = pca.singular_values_  # (numPC,)
        mixedfilters = PCs
        # mixedfilters = PCs.reshape((numPC, d1, d2)) # (numPC, d1, d2)
        mixedsig = dualPCAtransfer(data_normalized, PCs, single_values) # (numPC, T)

        covtrace = np.trace(pca.get_covariance()) / pixelNum

    if outputdir is not None:
        print("Saving PCA results to output directory not yet implemented")
    return mixedsig, mixedfilters, eigenvalues, covtrace, temporal_mean, spacial_mean


                   
CMP = plt.get_cmap('hot')
FIGSIZE = (6, 6)
def viewPCAResults(mixedfilters: np.ndarray, outputdir: Path | str,packN:Optional[int]=None):
    """
    Paint all PCs as heatmaps and save to outputdir.
    :param mixedfilters: shape (numPC, height, width)
    :param outputdir: directory to save images
    """
    outdir = Path(outputdir)
    outdir.mkdir(parents=True, exist_ok=True)

    numPC = mixedfilters.shape[0]
    for i in range(numPC):
        pc = mixedfilters[i]
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(pc, cmap=CMP, aspect='equal')
        ax.set_title(f"PC {i+1}")
        ax.axis('off')
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        fname = outdir / f"PC_{i+1:03d}.png"
        fig.tight_layout()
        fig.savefig(str(fname), dpi=150)
        plt.close(fig)
    if packN is not None and packN > 1:
        R = np.floor(np.sqrt(packN)).astype(int)
        C = np.ceil(packN / R).astype(int)
        fig,axs = plt.subplots(R, C,figsize=FIGSIZE)
        for i in range(R):
            for j in range(C):
                idx = i * C + j
                ax = axs[i, j]
                if idx < numPC:
                    pc = mixedfilters[idx]
                    img = ax.imshow(pc, cmap=CMP, aspect='equal')
                    # ax.set_title(f"PC {idx+1}")
                    ax.axis('off')
                else:
                    ax.axis('off')
        fig.tight_layout()
        fig.savefig(str(outdir / f"PCs_packed_{packN}.png"), dpi=300)
        

def tiff_info(fn: Path | str) -> Tuple[int, int, int]:
    img = imread(fn)
    if img.ndim == 2:
        h, w = img.shape
        nt = 1
    elif img.ndim == 3:
        nt, h, w = img.shape
    else:
        raise ValueError("Unsupported TIFF image dimensions")
    return h, w, nt


## TODO check noise definaition
## TODO detailize docstring
def plot_pc_spectrum(fn, cov_evals, pc_use=None, ax=None, show=True):
    """
    Python port of CellsortPlotPCspectrum(fn, CovEvals, PCuse)

    :param fn: path to TIFF movie file
    :param cov_evals: 1D 
    :param pc_use: 
    :param ax: 
    :param show: whether to call plt.show()

    :Return:
    :retype: Tuple[fig, ax]
    """
    cov_evals = np.asarray(cov_evals).astype(float).flatten()
    if cov_evals.size == 0:
        raise ValueError("cov_evals is empty")

    h, w, nt = tiff_info(fn)
    npix = int(h) * int(w)

    # Random matrix prediction (Sengupta & Mitra)
    p1 = npix
    q1 = nt
    q = max(p1, q1)
    p = min(p1, q1)
    sigma = 1.0
    lmax = sigma * np.sqrt(p + q + 2.0 * np.sqrt(p * q))
    lmin = sigma * np.sqrt(p + q - 2.0 * np.sqrt(p * q))
    lambdas = np.linspace(lmin, lmax, 1000)
    # Marchenko-Pastur-like density used in MATLAB code
    with np.errstate(invalid='ignore'):
        rho = (1.0 / (np.pi * lambdas * (sigma**2))) * np.sqrt(
            (lmax**2 - lambdas**2) * (lambdas**2 - lmin**2)
        )
    rho[np.isnan(rho)] = 0.0
    rhocdf = np.cumsum(rho)
    if rhocdf[-1] <= 0:
        rhocdf = np.linspace(0, 1, len(rho))
    else:
        rhocdf = rhocdf / rhocdf[-1]

    # interpolation targets: MATLAB used [p:-1:1]'/p
    targets = np.arange(p, 0, -1) / float(p)
    # ensure rhocdf is strictly increasing for np.interp: it is non-decreasing
    noise_lambda = np.interp(targets, rhocdf, lambdas)
    noiseigs = noise_lambda**2  # match MATLAB squaring

    # Normalize the PC spectrum as MATLAB did
    len_cov = cov_evals.size
    normrank = int(min(max(1, nt - 1), len_cov))  # ensure >=1
    # MATLAB indexing: noiseigs(normrank) and noiseigs(1)
    # convert to 0-based
    idx_norm = min(normrank, noiseigs.size) - 1
    denom_noise = noiseigs[0]
    if denom_noise == 0:
        denom_noise = 1.0
    scale = noiseigs[idx_norm] / denom_noise
    # Avoid division by zero for cov_evals[idx_norm]
    cov_ref = cov_evals[idx_norm] if cov_evals[idx_norm] != 0 else 1.0
    pca_norm = cov_evals * (scale / cov_ref)

    # Prepare plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ranks_pca = np.arange(1, len_cov + 1)
    ranks_noise = np.arange(1, noiseigs.size + 1)

    ax.plot(ranks_pca, pca_norm, 'o-', color=[0.3,0.3,0.3], markerfacecolor=[0.3,0.3,0.3], linewidth=2, label='Data variance')
    ax.plot(ranks_noise, noiseigs / denom_noise, 'b-', linewidth=2, label='Noise floor')
    ax.plot(ranks_noise, 2.0 * noiseigs / denom_noise, 'b--', linewidth=2, label='2 x Noise floor')

    if pc_use is not None and len(pc_use) > 0:  
        pc_use_arr = np.asarray(pc_use, dtype=int)
        valid = (pc_use_arr >= 0) & (pc_use_arr < len_cov)
        if np.any(valid):
            ax.plot(pc_use_arr[valid] + 1, pca_norm[pc_use_arr[valid]], 'rs', linewidth=2, label='Retained PCs')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('PC rank')
    ax.set_ylabel('Normalized variance')
    ax.set_title(os.path.basename(fn).replace('_', ' '))
    ax.legend()
    ax.grid(False)
    # format axes similar to MATLAB formataxes
    ax.tick_params(direction='out', length=6, width=1.5)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax