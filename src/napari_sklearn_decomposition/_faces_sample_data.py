from typing import List

from napari.types import LayerDataTuple


def faces_sample() -> List[LayerDataTuple]:
    """An example of a  Sample Data Function.

    Note: Sample Data with URIs don't need python code.
    """
    from numpy.random import RandomState
    from sklearn.datasets import fetch_olivetti_faces

    rng = RandomState(0)

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)

    return [(faces, {"name": "Olivetti Faces"})]
