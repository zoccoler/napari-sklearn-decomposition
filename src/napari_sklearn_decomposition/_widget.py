"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
import napari
import numpy as np
from magicgui import magic_factory, magicgui
from napari import Viewer
from qtpy.QtWidgets import QComboBox, QPushButton, QSpinBox, QVBoxLayout, QWidget


def linearize_image(image):
    shape = image.shape
    image_lin = image.reshape(shape[0], shape[1] * shape[2])
    return (image_lin, shape)


def image_reshape(image, n_components, shape):
    return image.reshape((n_components, shape[1], shape[2]))


@magic_factory()
def PCA(
    image: "napari.types.ImageData",
    n_components: int = 6,
    whiten: bool = True,
    svd_solver: str = "auto",
) -> "napari.types.ImageData":
    from sklearn.decomposition import PCA

    image, shape = linearize_image(image)
    pca = PCA(n_components=n_components)
    pca.fit(image)
    output_image = image_reshape(pca.components_, n_components, shape)
    print(output_image.shape)
    return output_image


@magic_factory()
def FastICA(
    image: "napari.types.ImageData", n_components: int = 6, whiten: bool = True
) -> "napari.types.ImageData":
    from sklearn.decomposition import FastICA

    image, shape = linearize_image(image)
    ica = FastICA(n_components=n_components)
    ica.fit(image)
    output_image = image_reshape(ica.components_, n_components, shape)
    print(output_image.shape)
    return output_image


@magic_factory()
def NMF(
    image: "napari.types.ImageData",
    n_components: int = 6,
    init: str = "warn",
    tol: float = 5e-3,
):
    from sklearn.decomposition import NMF

    image, shape = linearize_image(image)
    nmf = NMF(n_components=n_components, init=init, tol=tol)
    nmf.fit(image)
    output_image = image_reshape(nmf.components_, n_components, shape)
    print(output_image.shape)
    return output_image


def on_create(new_widget):
    # print('viewer = ', viewer)
    mapping = {"PCA": PCA, "NMF": NMF, "FastICA": FastICA}
    print("new_wid", new_widget)
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
        if len(new_widget) > 1:
            new_widget.pop()
        new_widget.append(chosen_widget)

    _on_choice_changed("PCA")


@magic_factory(
    choice={"choices": ["PCA", "NMF", "FastICA"]},
    call_button=False,
    widget_init=on_create,
)
def decomposition(choice: str, viewer: Viewer):

    # print('viewer = ', viewer)
    pass
