"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QSpinBox
from magicgui import magicgui, magic_factory
import napari
from napari import Viewer
import numpy as np

def linearize_image(image):
    shape = image.shape
    image_lin = image.reshape(shape[0], shape[1]*shape[2])
    return(image_lin, shape)

@magic_factory()
def PCA(image: 'napari.types.ImageData', n_components: int, whiten:bool=True, svd_solver:str='auto') -> 'napari.types.ImageData':
    from sklearn.decomposition import PCA

    image, shape = linearize_image(image)
    pca = PCA(n_components=n_components)
    pca.fit(image)
    output_image = pca.components_.reshape((n_components, shape[1], shape[2]))
    print(output_image.shape)
    return(output_image)

@magic_factory()
def NMF(n_components: int, init:str="nndsvda", tol:float=5e-3):
    from sklearn.decomposition import NMF
    image = np.arange(75).reshape(3,25)
    nmf = NMF(n_components=n_components, init=init, tol=tol)
    nmf.fit(image)
    return(nmf.noise_variance_.reshape(1, -1))

def on_create(new_widget):
    # print('viewer = ', viewer)
    mapping = {'PCA': PCA, 'NMF' : NMF}
    print('new_wid', new_widget)
    # new_widget.
    @new_widget.choice.changed.connect
    def _on_choice_changed(new_choice: str):

        # do whatever you need to create the widget, or look it up from some map
        # NOTE! consider the lifetime of the widget you are looking up and adding here.
        # you may wish to create it each time with `magicgui(some_function)` (see pattern below for tips)

        print(new_choice)
        # or you may wish to use a `magic_factory` instead
        factory = mapping[new_choice]
        chosen_widget = factory()
        print(new_widget)
        if len(new_widget)>1:
            new_widget.pop()
        new_widget.append(chosen_widget)
    _on_choice_changed('PCA')

@magic_factory(choice={'choices': ['PCA', 'NMF', 'FastICA']}, call_button=False,
               widget_init=on_create)
def decomposition(choice: str, viewer:Viewer):

    # print('viewer = ', viewer)
    pass



