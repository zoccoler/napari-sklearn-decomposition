import numpy as np

def linearize_image(image: np.ndarray) -> tuple:
    shape = image.shape
    image_lin = image.reshape(shape[0], shape[1] * shape[2])
    return (image_lin, shape)


def image_reshape(image: np.ndarray, n_components: int, shape: tuple) -> np.ndarray:
    return image.reshape((n_components, shape[1], shape[2]))


def set_small_mask(mask):
    opposite_mask = np.copy(~mask)
    if mask.sum() > opposite_mask.sum():
        return opposite_mask
    return mask
    
def filters_to_masks(space_filters):
    from skimage.filters import threshold_otsu, gaussian
    space_masks = np.zeros((space_filters.shape), dtype=int)
    for i in range(space_filters.shape[0]):
        space_filters_blur = gaussian(space_filters[i])
        th = threshold_otsu(space_filters_blur)
        
        binary_mask = space_filters_blur > th
        
        
        # binary_mask = set_small_mask(binary_mask)
        space_masks[i, binary_mask] += i + 1 
    return space_masks