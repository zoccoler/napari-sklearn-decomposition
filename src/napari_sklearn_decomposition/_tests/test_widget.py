from typing import Callable

import napari
import numpy as np
import pytest

from napari_sklearn_decomposition import decomposition
from napari_sklearn_decomposition._widget import NMF, PCA, FastICA


def test_plugin_widget_added(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer()
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-sklearn-decomposition", widget_name="Decomposition Widget"
    )
    assert len(viewer.window._dock_widgets) == 1


def test_widget_added(make_napari_viewer: Callable[..., napari.Viewer]) -> None:
    # Make a viewer with an image
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 50, 50)), name="test")

    widget = decomposition()
    viewer.window.add_dock_widget(widget)
    # Check widget was added
    assert len(viewer.window._dock_widgets) == 1
    # Check that the default choice is correct
    assert widget.choice.get_value() == "PCA"
    # Check that the PCA widget was created properly
    assert widget.PCA.n_components.get_value() == 6
    assert widget.PCA.svd_solver.get_value() == "auto"
    assert widget.PCA.whiten.get_value()
    assert (widget.PCA.image.get_value() == viewer.layers["test"].data).all()


@pytest.mark.parametrize(
    "method",
    [
        PCA,
        NMF,
        FastICA,
    ],
)
def test_decompositions(
    make_napari_viewer: Callable[..., napari.Viewer], method
) -> None:
    # Make a viewer with an image
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 50, 50)))
    # Launch the widget for the decomposition method
    widget = method()
    viewer.window.add_dock_widget(widget)
    # Run the widget with default settings
    widget()
    # Check a new image was added
    assert len(viewer.layers) == 2
    # Check that the name & shape of the new layer is correct
    assert viewer.layers[-1].name == f"{method.__name__} result"
    assert viewer.layers[-1].data.shape == (6, 50, 50)
    # Check that the colormap is correct
    # assert viewer.layers[-1].colormap.name == "PiYG"


# # make_napari_viewer is a pytest fixture that returns a napari viewer object
# # capsys is a pytest fixture that captures stdout and stderr output streams
# def test_example_q_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))

#     # create our widget, passing in the viewer
#     my_widget = ExampleQWidget(viewer)

#     # call our widget method
#     my_widget._on_click()

#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "napari has 1 layers\n"

# def test_example_magic_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))

#     # this time, our widget will be a MagicFactory or FunctionGui instance
#     my_widget = example_magic_widget()

#     # if we "call" this object, it'll execute our function
#     my_widget(viewer.layers[0])

#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == f"you have selected {layer}\n"
