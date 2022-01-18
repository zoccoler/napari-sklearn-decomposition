"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html

Replace code below according to your needs.
"""
from __future__ import annotations

from napari.types import LayerDataTuple


def faces_sample() -> LayerDataTuple:
    from numpy.random import RandomState
    from sklearn.datasets import fetch_olivetti_faces

    rng = RandomState(0)

    faces = fetch_olivetti_faces(return_X_y=False, shuffle=True, random_state=rng)

    return [(faces.images, {"name": "Olivetti Faces"})]
