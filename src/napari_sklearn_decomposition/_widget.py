"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from typing import List

if TYPE_CHECKING:
    import napari.types
from napari.types import LayerDataTuple

from .utils import filters_to_masks, linearize_image, image_reshape


def add_plot_widget():
    from ._plotting.features_line_plot import FeaturesLinePlotWidget
    viewer = napari.current_viewer()
    plot_widget = FeaturesLinePlotWidget(viewer,call_button=False)
    if 'napari-sklearn-decomposition: Temporal Components' not in \
        viewer.window._dock_widgets.data.keys():
            viewer.window.add_dock_widget(plot_widget,
                name = 'napari-sklearn-decomposition: Temporal Components',
                add_vertical_stretch = True)
    for layer in viewer.layers:
        if 'Label Component' in layer.name:
            layer.contour = 1 

@magic_factory()
def PCA_napari(
    image: "napari.types.ImageData",
    n_components: int = 6,
    whiten: bool = True,
    svd_solver: str = "auto",
    as_labels: bool = True,
) -> "napari.types.LayerDataTuple":
    from sklearn.decomposition import PCA
    @PCA_napari.called.connect
    def call_plot():
        add_plot_widget()
    
    image_lin, shape = linearize_image(image)
    pca = PCA(n_components=n_components)
    pca.fit(image_lin)
    output_image = image_reshape(pca.components_, n_components, shape)
    output_labels, _ = filters_to_masks(output_image)
    # IF TIME SERIES
    # Time signals are just average over time using labels
    time_signals = np.asarray([np.mean(image[:, output_labels[i].astype(bool)], axis=1) for i in range(n_components)]).T
    viewer = napari.current_viewer()
    viewer.dims.set_point(0, 0)
    if as_labels is True:
        layer_list = [(output_labels[i],
                       {"name": "PCA Label Component #" + str(i),
                        # "_contour": 1,
                        "features": pd.DataFrame(time_signals[:,i], 
                                                 columns=['Component #' \
                                                          + str(i)])},
                       "labels") for i in range(n_components)]
    else:
        layer_list = [(output_image[i],
                       {"name": "PCA Eigenvector #" + str(i),
                        "colormap": "PiYG",
                        "metadata": {'Component #' + str(i): time_signals[:,i]}
                        },
                       "image") for i in range(n_components)]
    
    return layer_list


@magic_factory()
def FastICA_napari(
    image: "napari.types.ImageData",
    n_components: int = 6,
    as_labels: bool = True,
    whiten: str = "arbitrary-variance",
    fun: str = 'logcosh',
    max_iter: int = 200,
    tol: int = -6,
    w_init: np.array = None,
    random_state = None
) -> "napari.types.LayerDataTuple":
    from sklearn.decomposition import FastICA
    from scipy.stats import skew
    @FastICA_napari.called.connect
    def call_plot():
        add_plot_widget()
    
    image_lin, shape = linearize_image(image)
    ica = FastICA(n_components=n_components, whiten = whiten, fun = fun,
                  max_iter = max_iter, tol = 10**(tol), w_init = w_init,
                  random_state = random_state)
    ica.fit(image_lin)
    
    components = ica.components_
    # Sort components by skewness (decreasing order)
    components_skewness = skew(components, axis=1)[::-1]
    skew_sort_indices = np.argsort(abs(components_skewness))
    components = components[skew_sort_indices]
    
    output_image = image_reshape(components, n_components, shape)
    output_labels, _ = filters_to_masks(output_image)
    # IF TIME SERIES
    # Time signals are just average over time using labels
    time_signals = np.asarray([np.mean(image[:, output_labels[i].astype(bool)], axis=1) for i in range(n_components)]).T
    
    viewer = napari.current_viewer()
    viewer.dims.set_point(0, 0)
    if as_labels is True:
        layer_list = [(output_labels[i],
                       {"name": "ICA Label Component #" + str(i),
                        # "_contour": 1,
                        "features": pd.DataFrame(time_signals[:,i], 
                                                 columns=['Component #' \
                                                          + str(i)])},
                       "labels") for i in range(n_components)]
    else:
        layer_list = [(output_image[i],
                       {"name": "ICA Component #" + str(i),
                        "colormap": "PiYG",
                        "metadata": {'Component #' + str(i): time_signals[:,i]}
                        },
                       "image") for i in range(n_components)]
    
    return layer_list


@magic_factory()
def NMF_napari(
    image: "napari.types.ImageData",
    n_components: int = 6,
    init: str = "nndsvda",
    tol: float = 5e-3,
    as_labels: bool = True,
) -> "napari.types.LayerDataTuple":
    from sklearn.decomposition import NMF
    from scipy.stats import skew
    @NMF_napari.called.connect
    def call_plot():
        add_plot_widget()
    
    image_lin, shape = linearize_image(image)
    nmf = NMF(n_components=n_components, init=init, tol=tol)
    nmf.fit(image_lin)
    
    components = nmf.components_
    # Sort components by skewness (decreasing order)
    components_skewness = skew(components, axis=1)[::-1]
    skew_sort_indices = np.argsort(abs(components_skewness))
    components = components[skew_sort_indices]
    
    output_image = image_reshape(components, n_components, shape)
    output_labels, _ = filters_to_masks(output_image)
    # IF TIME SERIES
    # Time signals are just average over time using labels
    time_signals = np.asarray([np.mean(image[:, output_labels[i].astype(bool)], axis=1) for i in range(n_components)]).T
    
    viewer = napari.current_viewer()
    viewer.dims.set_point(0, 0)
    if as_labels is True:
        layer_list = [(output_labels[i],
                       {"name": "NMF Label Component #" + str(i),
                        # "_contour": 1,
                        "features": pd.DataFrame(time_signals[:,i], 
                                                 columns=['Component #' \
                                                          + str(i)])},
                       "labels") for i in range(n_components)]
    else:
        layer_list = [(output_image[i],
                       {"name": "NMF Component #" + str(i),
                        "colormap": "viridis",
                        "metadata": {'Component #' + str(i): time_signals[:,i]}
                        },
                       "image") for i in range(n_components)]
    
    return layer_list

@magic_factory(tol = {"label": "tol 10^ :"})
def stICA_napari(
    image: "napari.types.ImageData", n_components: int = 6, mu: float = 0.2,
    as_labels: bool = True, whiten: str = "arbitrary-variance", fun: str = 'logcosh',
    max_iter: int = 200, tol: int = -6, random_state = None
) -> List[LayerDataTuple]:
    from .stICA import stICA

    
    @stICA_napari.called.connect
    def call_plot():
        add_plot_widget()
            
    
    space_filters, time_signals = stICA(image = image, mu = mu, 
                                        n_components = n_components,
                                        as_labels = as_labels,
                                        random_state = random_state,
                                        whiten = whiten, fun = fun, 
                                        max_iter = max_iter, tol = 10**(tol))
    
    space_filters = [space_filters[i] for i in range(space_filters.shape[0])]
    viewer = napari.current_viewer()
    viewer.dims.set_point(0, 0)
    if as_labels is True:
        layer_list = [(space_filters[i],
                       {"name": "stICA Label Component #" + str(i),
                        # "_contour": 1,
                        "features": pd.DataFrame(time_signals[:,i], 
                                                 columns=['Component #' \
                                                          + str(i)])},
                       "labels") for i in range(n_components)]
    else:
        layer_list = [(space_filters[i],
                       {"name": "stICA Space Filter #" + str(i),
                        "colormap": "PiYG",
                        "metadata": {'Component #' + str(i): time_signals[:,i]}
                        },
                       "image") for i in range(n_components)]
    
    return layer_list    

def on_create(new_widget):
    mapping = {"PCA": PCA_napari, "NMF": NMF_napari, "FastICA": FastICA_napari, "stICA": stICA_napari}
    # print("new_wid", new_widget)
    # new_widget.

    @new_widget.choice.changed.connect
    def _on_choice_changed(new_choice: str):

        # do whatever you need to create the widget, or look it up from some map
        # NOTE! consider the lifetime of the widget you are looking up and adding here.
        # you may wish to create it each time with `magicgui(some_function)`
        # (see pattern below for tips)

        # print(new_choice)
        # or you may wish to use a `magic_factory` instead
        factory = mapping[new_choice]
        chosen_widget = factory()
        # print(new_widget)
        if len(new_widget) > 1:
            new_widget.pop()
        new_widget.append(chosen_widget)
        # reset_choices from:
        # https://github.com/pattonw/napari-affinities/blob/7d9eab9100daf607ce7351f8717bff37ad71acf0/src/napari_affinities/widget.py#L61
        chosen_widget.reset_choices()
        viewer = napari.current_viewer()
        viewer.layers.events.inserted.connect(chosen_widget.reset_choices)

    _on_choice_changed("PCA")


@magic_factory(
    choice={"choices": ["PCA", "NMF", "FastICA", "stICA"]},
    call_button=False,
    widget_init=on_create,
)
def decomposition(choice: str):
    pass
