"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QSpinBox
from magicgui import magicgui, magic_factory
import numpy as np
@magic_factory()
def PCA(n_components: int, whiten:bool=True, svd_solver:str='auto'):
    from sklearn.decomposition import PCA
    image = np.arange(25).reshape(5,5).ravel()
    pca = PCA(n_components=n_components)
    pca.fit(image)

    return(pca.noise_variance_.reshape(1, -1))

@magic_factory()
def NMF(n_components: int, init:str="nndsvda", tol:float=5e-3):
    from sklearn.decomposition import NMF
    image = np.arange(25).reshape(5,5).ravel()
    nmf = NMF(n_components=n_components, init=init, tol=tol)
    nmf.fit(image)
    return(nmf.noise_variance_.reshape(1, -1))

@magic_factory(choice={'choices': ['PCA', 'NMF', 'FastICA']}, call_button=False)
def decomposition(choice: str):
    pass

# mapping = {'PCA': PCA}

@decomposition.choice.changed.connect
def _on_choice_changed(new_choice: str):
    # do whatever you need to create the widget, or look it up from some map
    # NOTE! consider the lifetime of the widget you are looking up and adding here.
    # you may wish to create it each time with `magicgui(some_function)` (see pattern below for tips)

    print(new_choice)
    # or you may wish to use a `magic_factory` instead
    # factory = mapping[new_choice]
    # new_widget = factory()
    # decomposition.pop()
    # decomposition.append(new_widget)