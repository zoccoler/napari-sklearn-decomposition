"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox
from magicgui import magic_factory



class DecompositionQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Run decomposition")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

        # add QComboBox to the layout
        self.decomposition_type = QComboBox()
        self.decomposition_type.addItems(["PCA", "ICA", "FA", "NMF"])
        self.layout().addWidget(self.decomposition_type)

        # add button connection

        # add QSpinBox for number of components

        # add other options with display in/out

        # add callack to add the decomposed image to viewer

        # think how to display images

        # add dataset from sklearn as sample data



    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")