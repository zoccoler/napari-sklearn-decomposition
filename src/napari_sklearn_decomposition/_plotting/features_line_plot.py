import napari
from magicgui import magicgui
import numpy as np
import pandas as pd
from napari_matplotlib import ScatterWidget
from napari_matplotlib.util import Interval
from magicgui.widgets import ComboBox
from typing import List, Optional, Tuple
from qtpy.QtWidgets import QFileDialog
from napari.utils.colormaps import label_colormap

class FeaturesLinePlotWidget(ScatterWidget):
    """Plot widget to display histogram of selected layer features."""

    n_layers_input = Interval(1, 1)
    # Image and Labels layer
    input_layer_types = (
        napari.layers.Image,
        napari.layers.Labels,
        # napari.layers.Points,
        # napari.layers.Shapes,
        # napari.layers.Tracks,
        # napari.layers.Vectors,
    )

    def __init__(self, napari_viewer: napari.viewer.Viewer,
                 call_button: bool = False):
        super().__init__(napari_viewer)
        self.setMinimumSize(200, 200)
        self._key_selection_widget = magicgui(
            self._set_axis_keys,
            y_axis_key={"choices": self._get_valid_axis_keys},
            # n_bins={"value": 50, "widget_type": "SpinBox"},
            # auto_call=auto_call,
            call_button=call_button
        )

        # if self._key_selection_widget._auto_call==False:
        #     self._key_selection_widget._call_button = "Update"
        self._export_button = magicgui(
            self.export,
            call_button='Export plot as csv'
            )
        self.layout().addWidget(self._key_selection_widget.native)
        self.layout().addWidget(self._export_button.native)
        self._on_update_layers()

    @property
    def y_axis_key(self) -> Optional[str]:
        """Key to access y axis data from the FeaturesTable"""
        return self._y_axis_key

    @y_axis_key.setter
    def y_axis_key(self, key: Optional[str]) -> None:
        self._y_axis_key = key
        self._draw()

    def _set_axis_keys(self, y_axis_key: str) -> None:
        """Set both axis keys and then redraw the plot."""
        self._y_axis_key = y_axis_key
        # self._n_bins = n_bins
        self._draw()

    def _get_valid_axis_keys(
        self, combo_widget: Optional[ComboBox] = None
    ) -> List[str]:
        """
        Get the valid axis keys from the layer FeatureTable.

        Returns
        -------
        axis_keys : List[str]
            The valid axis keys in the FeatureTable. If the table is empty
            or there isn't a table, returns an empty list.
        """
        self.valid_metadata_keys = []
        if len(self.layers) == 0:
            return []
        elif isinstance(self.layers[0], napari.layers.Image) \
            and self.layers[0].metadata:
            # Return metadata keys if and where 'Component' is present
            component_indices = []
            # Look for 'Component' in metadata keys and store its indices
            for i, key_value in enumerate(self.layers[0].metadata.items()):
                key, value = key_value
                if 'Component' in key:
                    component_indices.append(i)
            if component_indices:
                # Return list of keys containing 'Component'
                self.valid_metadata_keys = [list(self.layers[0].metadata.keys())[i] for i in component_indices]
                return self.valid_metadata_keys
            else:
                return []

        elif isinstance(self.layers[0], napari.layers.Image) \
            and not self.layers[0].metadata:
            return []
        else:
            return list(self.layers[0].features.keys())

    def _get_data(self) -> Tuple[List[np.ndarray], str, int]:
        """Get the plot data.

        Returns
        -------
        data : List[np.ndarray]
            List contains X and Y columns from the FeatureTable. Returns
            an empty array if nothing to plot.
        x_axis_name : str
            The title to display on the x axis. Returns
            an empty string if nothing to plot.
        y_axis_name: int
            The title to display on the y axis. Returns
            an empty string if nothing to plot.
        """
        
        if isinstance(self.layers[0], napari.layers.Image):
            valid_metadata = {k:v for k,v in self.layers[0].metadata.items()
                              if k in self.valid_metadata_keys}
            feature_table = pd.DataFrame(valid_metadata)
        else:
            feature_table = self.layers[0].features
        if (
            (len(feature_table) == 0)
            or (self.y_axis_key is None)
        ):
            return [], "", ""

        data_y = feature_table[self.y_axis_key]
        data_x = np.arange(len(data_y))
        data = [data_x, data_y]

        x_axis_name = 'Frame/Sample #'
        y_axis_name = self.y_axis_key.replace("_", " ") + ' (A.U.)'

        return data, x_axis_name, y_axis_name

    def _on_update_layers(self) -> None:
        """This is called when the layer selection changes by
        ``self.update_layers()``.

        """
        
        if hasattr(self, "_key_selection_widget"):
            self._key_selection_widget.reset_choices()
            
            # if a layer is selected
            if self.layers:
                # if selected layer has features or metadata
                if hasattr(self.layers[0], "features") or self.layers[0].metadata:
                    # get feature key/name and plot it
                    self._get_valid_axis_keys(self._key_selection_widget.y_axis_key)
                    self._set_axis_keys(self._key_selection_widget.y_axis_key.current_choice)
                    self.draw()
        else:
            # reset the axis keys
            self._x_axis_key = None
            self._y_axis_key = None
        
                

    def draw(self) -> None:
        """Clear the axes and histogram the currently selected layer/slice."""
        data, x_axis_name, y_axis_name = self._get_data()

        if len(data) == 0:
            # don't plot if there isn't data
            return
        if isinstance(self.layers[0], napari.layers.Labels):
            color = self.layers[0].get_color(self.layers[0].data.max())
        else:
            color_id = int(self.y_axis_key[-1]) + 1
            color = label_colormap(50, 0.5).colors[color_id]
        self.line, = self.axes.plot(data[0], data[1], alpha=self._marker_alpha,
                                   color=color)
        running_dot = self.axes.plot(data[0][self.current_z],
                                     data[1][self.current_z],
                                     '*', color='white')
        self.axes.set_xlabel(x_axis_name)
        self.axes.set_ylabel(y_axis_name)

    def export(self) -> None:
        """Export plotted data as csv."""
        if not self.axes.has_data():
            return
        # Not including last bin because len(bins) = len(N) + 1
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
        df_to_save = pd.DataFrame({self.axes.get_xlabel(): self.line.get_data()[0],
                                   self.axes.get_ylabel(): self.line.get_data()[1]})
        fname = QFileDialog.getSaveFileName(self, 'Save plotted data',
                                            'c:\\',
                                            "Csv files (*.csv)")
        df_to_save.to_csv(fname[0])
        return
