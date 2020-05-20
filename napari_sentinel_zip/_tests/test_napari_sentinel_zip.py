import numpy as np
from napari_sentinel_zip import napari_get_reader
import os
import napari.components

TEST_TILE_PATH = os.path.abspath("./napari_sentinel_zip/_tests/Test_Tile")

TEST_ONE_ZIP_PATH = os.path.abspath(
    "./napari_sentinel_zip/_tests/Test_Tile/SENTINEL2A_20171008-002414-496_L2A_T55HBU_C_V1-0.zip"
    )

for root, dirnames, filenames in os.walk(TEST_TILE_PATH):
    TEST_ZIP_PATH_LIST = [os.path.join(root, filename) for filename in filenames]


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


def check_layer_list_structure(layer_list, num_ims, num_layers):
    # layer list must be list
    assert isinstance(layer_list, list), "reader does not return list"

    # layer list must have expected number of layers
    assert len(layer_list) == num_layers, f"Layer list has {len(layer_list)} layers, expected {NUM_LAYERS}"

    for i in range(len(layer_list)):
        layer = layer_list[i]
        # all elements in the layers must be tuples
        assert isinstance(layer, tuple), f"Layer list element {i} is not tuple"

            # each data in tuples must have correct shape
        expected_shape = tuple([num_ims] + [im_axis_shape for im_axis_shape in IM_SHAPES[i]])
        assert layer[0].shape == expected_shape,\
        f"Layer {i} has shape {layer[0].shape}, expected {expected_shape}"


def check_viewer_open_structure(path, num_layers, num_dims):
    # napari must be able to open the path directly
    v = napari.components.ViewerModel()
    v.open(path)

    # viewer must have 21 layers after opening
    assert len(v.layers) == num_layers, f"Expected {num_layers} layers after opening in napari viewer, got {len(v.layers)}"
    # viewer should have 3 dimensions after opening
    assert v.dims.ndim == num_dims, f"Expected {num_dims} dimensions after opening in napari viewer, got {v.dims.ndim}"

def test_reader_with_list():
    NUM_IMS = 10
    NUM_LAYERS = 21
    NUM_DIMS = 3

    reader = napari_get_reader(TEST_ZIP_PATH_LIST)

    layer_list = reader(TEST_ZIP_PATH_LIST)

    check_layer_list_structure(layer_list, NUM_IMS, NUM_LAYERS)

    v = napari.components.ViewerModel()
    for layer in layer_list:
        v.add_image(layer[0], **layer[1])

    # viewer must have 21 layers after opening
    assert len(v.layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers after opening in napari viewer, got {len(v.layers)}"
    # viewer should have 3 dimensions after opening
    assert v.dims.ndim == NUM_DIMS, f"Expected {NUM_DIMS} dimensions after opening in napari viewer, got {v.dims.ndim}"
   

def test_reader_with_string_path():
    NUM_IMS = 1
    NUM_LAYERS = 21
    NUM_DIMS = 3

    reader = napari_get_reader(TEST_ONE_ZIP_PATH)
    
    layer_list = reader(TEST_ONE_ZIP_PATH)

    check_layer_list_structure(layer_list, NUM_IMS, NUM_LAYERS)
    
    check_viewer_open_structure(TEST_ONE_ZIP_PATH, NUM_LAYERS, NUM_DIMS)


def test_reader_with_root_directory():
    NUM_IMS = 10
    NUM_LAYERS = 21
    NUM_DIMS = 3

    reader = napari_get_reader(TEST_TILE_PATH)

    # get result of reader(path)
    layer_list = reader(TEST_TILE_PATH)

    check_layer_list_structure(layer_list, NUM_IMS, NUM_LAYERS)

    check_viewer_open_structure(TEST_TILE_PATH, NUM_LAYERS, NUM_DIMS)


def test_get_reader_invalid_input_returns_None():
    # path which is not a zip and doesn't contain zips
    reader = napari_get_reader("fake.file")
    assert reader is None

    # path which is a zip but not a SENTINEL zip
    reader = napari_get_reader("fake.zip")
    assert reader is None

    # list of non-sentinel zips
    reader = napari_get_reader(["fake.zip", "fake2.zip"])
    assert reader is None


def test_get_reader_valid_input_returns_reader():
    # list of sentinel zips
    reader = napari_get_reader(TEST_ZIP_PATH_LIST)
    assert callable(reader), "Passing list of Sentinel zips does not return valid reader"

    # solitary sentinel zip
    reader = napari_get_reader(TEST_ONE_ZIP_PATH)
    assert callable(reader), "Passing one Sentinel zip does not return valid reader"

    # root dir of zips
    reader = napari_get_reader(TEST_TILE_PATH)
    assert callable(reader), f"napari_get_reader() did not return a function"