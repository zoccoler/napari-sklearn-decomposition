from napari_sklearn_decomposition import faces_sample


def test_sample_data():
    # Run faces_sample() and compare the output to the expected output.
    faces = faces_sample()
    assert faces[0][0].shape == (400, 64, 64)
