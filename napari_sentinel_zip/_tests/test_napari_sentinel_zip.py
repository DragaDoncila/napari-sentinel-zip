import numpy as np
from napari_sentinel_zip import napari_get_reader
import os
import napari.components

TEST_TILE_PATH = os.path.abspath("./napari_sentinel_zip/_tests/Test_Tile")

IM_SHAPES = [
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (10980, 10980),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (10980, 10980),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (5490, 5490),
    (1000, 1000, 3) #QKL
]

NUM_IMS = 10
NUM_LAYERS = 21
NUM_DIMS = 3

def test_reader():
    # get reader function using test zips
    reader = napari_get_reader(TEST_TILE_PATH)

    # reader must be callable
    assert callable(reader), f"napari_get_reader() did not return a function"

    # get result of reader(path)
    layer_list = reader(TEST_TILE_PATH)

    # resulting layers must be a list
    assert isinstance(layer_list, list), f"reader does not return list"

    for i in range(len(layer_list)):
        layer = layer_list[i]
        # all elements in the layers must be tuples
        assert isinstance(layer, tuple), f"Layer list element {i} is not tuple"

        # each data in tuples must have correct shape
        expected_shape = tuple([NUM_IMS] + [im_axis_shape for im_axis_shape in IM_SHAPES[i]])
        assert layer[0].shape == expected_shape,\
        f"Layer {i} has shape {layer[0].shape}, expected {expected_shape}"

    # napari must be able to open the path directly
    v = napari.components.ViewerModel()
    v.open(TEST_TILE_PATH)

    # viewer must have 21 layers after opening
    assert len(v.layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers after opening in napari viewer, got {len(v.layers)}"
    # viewer should have 3 dimensions after opening
    assert v.dims.ndim == NUM_DIMS, f"Expected {NUM_DIMS} dimensions after opening in napari viewer, got {v.dims.ndim}"

def test_get_reader_pass():
    # path which is not a zip and doesn't contain zips
    reader = napari_get_reader("fake.file")
    assert reader is None

    # path which is a zip but not a SENTINEL zip
    reader = napari_get_reader("fake.zip")
    assert reader is None

    # list of non-sentinel zips
    reader = napari_get_reader(["fake.zip", "fake2.zip"])
    assert reader is None