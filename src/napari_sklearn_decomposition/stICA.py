import numpy as np
from sklearn import decomposition
from .utils import filters_to_masks

def sPCA(image, n_components=None, random_state=None):
    
    image_ravel = np.asarray([image[i].ravel(order='F') for i in range(image.shape[0])])
    
    n_samples, n_pixels = image_ravel.shape
    # Center data
    image_ravel_centered = image_ravel - image_ravel.mean(axis=0) # subtract each pixel time average from each pixel
    image_ravel_centered -= image_ravel_centered.mean(axis=1).reshape(n_samples, -1) # Subtract each image spatial average from each image
    
    single_image_shape = (image.shape[-2], image.shape[-1])
    
    # PCA
    pca_estimator = decomposition.PCA(n_components = n_components, svd_solver = "randomized", whiten = True, random_state=random_state)
    pca_estimator.fit(image_ravel_centered)
    space_components_ravel = pca_estimator.components_ # spatial filters in 1D
    if n_components is None:
        n_components = image.shape[0]
    
    space_components = space_components_ravel.reshape(tuple([n_components, *single_image_shape]), order='F')
    
    # # In case you need the filtered movies, you have to apply transform
    # image_ravel_transformed = pca_estimator.transform(image_ravel_centered) # uncorrelated samples (timepoints)
    # # And apply the inverse transform (if n_components was not defined, you get back the same original video)
    # image_ravel_filtered = pca_estimator.inverse_transform(image_ravel_transformed)
    # print(image_ravel_filtered.shape)
    # image_filtered = image_ravel_filtered.reshape(image.shape, order='F')
    
    return(space_components)

def tPCA(image, n_components=None, random_state=None):
    image_ravel = np.asarray([image[i].ravel(order='F') for i in range(image.shape[0])]).T

    n_samples, n_timepoints  = image_ravel.shape
    
    # Center data
    image_ravel_centered = image_ravel - image_ravel.mean(axis=0) # subtract each pixel time average from each pixel
    image_ravel_centered -= image_ravel_centered.mean(axis=1).reshape(n_samples, -1) # Subtract each image spatial average from each image
    
    single_image_shape = (image.shape[-2], image.shape[-1])
    
    # PCA
    pca_estimator = decomposition.PCA(n_components = n_components, svd_solver = "randomized", whiten = True, random_state=random_state)
    pca_estimator.fit(image_ravel_centered)
    time_components = pca_estimator.components_ # time components (1D)
    
    # # In case you need the filtered movies, you have to apply transform
    # image_ravel_transformed = pca_estimator.transform(image_ravel_centered) # uncorrelated samples (pixels)
    # print(image_ravel_transformed.shape)
    # # And apply the inverse transform (if n_components was not defined, you get back the same original video)
    # image_ravel_filtered = pca_estimator.inverse_transform(image_ravel_transformed)
    # print(image_ravel_filtered.shape)
    # image_filtered = image_ravel_filtered.T.reshape(image.shape, order='F')
    
    return(time_components)


def stICA(image, mu, n_components=None, as_labels = False, random_state=None, **kwargs):
    from scipy.stats import skew
    space_filters, time_signals = None, None
    # Get 2D image shape
    single_image_shape = (image.shape[-2], image.shape[-1])
    # sPCA
    space_components = sPCA(image, n_components = n_components, random_state = random_state)
    space_components = np.asarray([space_components[i].ravel(order='F') for i in range(n_components)])
    n_px = space_components.shape[-1]
    # tPCA
    time_components = tPCA(image, n_components = n_components, random_state = random_state)
    n_t = time_components.shape[-1]
    
    ## Spatial-temporal ICA
    # Concatenate weighted space and time
    if mu==0: # Only space ICA
        sig_use = space_components
    elif mu==1: # Only time ICA
        sig_use = time_components
    else:
        sig_use = np.concatenate(((1-mu)*space_components, mu*time_components), axis=1)
    ica_estimator = decomposition.FastICA(random_state = random_state, **kwargs)
    S_ = ica_estimator.fit_transform(sig_use.T)  # Reconstruct signals
    A_ = ica_estimator.mixing_  # Get estimated mixing matrix
    
    # Sort components by skewness
    S_skewness = skew(S_, axis=0)[::-1]
    skew_sort_indices = np.argsort(abs(S_skewness))
    S_ = S_[:,skew_sort_indices]
    
    # Rebuild components to original shape
    if mu<1: # Space was considered
        space_filters = S_[:n_px].T
        space_filters = space_filters.reshape(tuple([n_components, *single_image_shape]), order = 'F')
        if as_labels == True:
            space_filters = filters_to_masks(space_filters)
    if mu>0 and mu !=1: # Time was also considered
        time_signals = S_[n_px:]
    elif mu==1: # Only time was considered
        time_signals = S_
    
    return space_filters, time_signals