__version__ = '0.2.1'

"""
This module stacks and loads Sentinel ZIP images into napari.
"""
import numpy as np
import re
import os
import tifffile
import zipfile
import dask
import dask.array as da
from napari_plugin_engine import napari_hook_implementation
from glob import glob
from collections import defaultdict
from skimage.io import imread

SENTINEL_PATH_REGEX = re.compile(r".*SENTINEL.*_[0-9]{8}.*\.zip")

# each zip file contains many bands, ie channels
BANDS = [
    "FRE_B11",
    "FRE_B12",
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B5",
    "FRE_B6",
    "FRE_B7",
    "FRE_B8",
    "FRE_B8A",
    "SRE_B11",
    "SRE_B12",
    "SRE_B2",  # surface reflectance, blue
    "SRE_B3",  # surface reflectance, green
    "SRE_B4",  # surface reflectance, red
    "SRE_B5",
    "SRE_B6",
    "SRE_B7",
    "SRE_B8",
    "SRE_B8A",
    ]

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
]

SCALES = np.concatenate([np.ones((len(IM_SHAPES), 1)), 10980 * 10 / np.array(IM_SHAPES)], axis=1)  # 10m per pix is highest res
OFFSETS = [(5, 5) if shape[0] == 5490 else (0, 0) for shape in IM_SHAPES]
SHAPES = dict(zip(BANDS, IM_SHAPES))
OFFSETS = dict(zip(BANDS, OFFSETS))
SCALES = dict(zip(BANDS, SCALES))

# add scales for the mask layers
MASK_SHAPES = [
    (10980, 10980),
    (5490, 5490),
]
SCALES['EDG_R1'] = (1, 10, 10)
SCALES['EDG_R2'] = (1, 20, 20)

CONTRAST_LIMITS = [-1000, 19_000]
QKL_SCALE = (1, 109.8, 109.8)

@dask.delayed
def ziptiff2array(zip_filename, path_to_tiff):
    """Return a NumPy array from a TiffFile within a zip file.

    Parameters
    ----------
    zip_filename : str
        Path to the zip file containing the tiff.
    path_to_tiff : str
        The path to the TIFF file within the zip archive.

    Returns
    -------
    image : numpy array
        The output image.

    Notes
    -----
    This is a delayed function, so it actually returns a dask task. Call
    ``result.compute()`` or ``np.array(result)`` to instantiate the data.
    """
    with zipfile.ZipFile(zip_filename) as zipfile_obj:
        open_tiff_file = zipfile_obj.open(path_to_tiff)
        tiff_f = tifffile.TiffFile(open_tiff_file)
        image = tiff_f.pages[0].asarray()
    return image

def sort_timestamps(path_list):
    timestamp_regex = re.compile(".*([0-9]{8}-[0-9]{6}-[0-9]{3}).*.zip")
    timestamp_dict = {}
    for path in path_list:
        match = timestamp_regex.match(path)
        if match:
            timestamp = match.groups()[0]
            timestamp_dict[timestamp] = path
    sorted_paths = []
    for timestamp in sorted(timestamp_dict.keys()):
        sorted_paths.append(timestamp_dict[timestamp])
    return sorted_paths

@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # if we've been handed a list of SENTINEL zips, that's fine
    if isinstance(path, list):
        # all paths within must be sentinel zips
        for pth in path:
            if not SENTINEL_PATH_REGEX.match(pth):
                return None
        return reader_function

    # if we've been handed a single zip that's fine
    if isinstance(path, str) and SENTINEL_PATH_REGEX.match(path):
        return reader_function
    
    # if we've been handed a root directory with SENTINEL zips inside, that's fine
    all_zips = glob(path + '/*.zip')
    filtered_zips = list(filter(SENTINEL_PATH_REGEX.match, all_zips))
    if not all_zips:
        return None

    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # one sentinel zip
    if isinstance(path, str) and SENTINEL_PATH_REGEX.match(path):
        paths = [path]
    # list of sentinel zips
    elif isinstance(path, list):
        paths = path
    # one root directory path with multiple sentinel zips inside
    else:
        paths = list(filter(SENTINEL_PATH_REGEX.match, glob(path + "/*.zip")))

    paths = sort_timestamps(paths)
    
    # stack all timepoints together for each band
    images = {}
    for band, shape in zip(BANDS, IM_SHAPES):
        stack = []
        for fn in paths:
            # get all the tiffs
            basepath = os.path.splitext(os.path.basename(fn))[0]
            path = basepath + '/' + basepath + '_' + band + '.tif'
            image = da.from_delayed(
                ziptiff2array(fn, path), shape=shape, dtype=np.int16
            )
            stack.append(image)

        images[band] = da.stack(stack)
    
    # get the edge masks
    masks = {}
    for idx, shape in zip((1, 2), MASK_SHAPES):
        stack = []
        for fn in paths:
            basepath = os.path.splitext(os.path.basename(fn))[0]
            path = basepath + '/MASKS/' + basepath + f'_EDG_R{idx}.tif'
            image = da.from_delayed(
                ziptiff2array(fn, path), shape=shape, dtype=np.uint8
            )
            stack.append(image)
        masks[f'EDG_R{idx}'] = da.stack(stack)

    # get the quicklook jpg
    jpg_stack = []
    for fn in paths:
        basepath = os.path.splitext(os.path.basename(fn))[0]
        path = basepath + '/' + basepath + '_' + 'QKL_ALL.jpg'
        zip_obj = zipfile.ZipFile(fn)
        open_jpg = zip_obj.open(path)
        image = imread(open_jpg)
        jpg_stack.append(image)
    jpg_im = da.stack(jpg_stack)

    # decide on colourmap
    colormaps = defaultdict(lambda: 'gray')
    for band in BANDS:
        if band.endswith('B2'):
            colormaps[band] = 'blue'
        elif band.endswith('B3'):
            colormaps[band] = 'green'
        elif band.endswith('B4'):
            colormaps[band] = 'red'


    layer_list = []
    layer_type = "image"
    add_kwargs = {
        "name": 'QKL_ALL',
        "multiscale": False,
        "scale": QKL_SCALE,
        "rgb": True,
        "visible": True,
        "contrast_limits": CONTRAST_LIMITS
    }
    layer_list.append((jpg_im, add_kwargs, layer_type))

    for mask_band, mask in masks.items():
        add_kwargs = {
            "name": mask_band,
            "scale": SCALES[mask_band],
            "visible": False,
        }
        layer_list.append((mask, add_kwargs, 'labels'))
    
    for band, image in images.items():
        colormap = colormaps[band]
        blending = 'additive' if colormaps[band] != 'gray' else 'translucent'
        add_kwargs = {
            "name": band,
            "multiscale": False,
            "scale": SCALES[band],
            "colormap": colormap,
            "blending": blending,
            "visible": False,
            "contrast_limits": CONTRAST_LIMITS
        }
        layer_list.append((image, add_kwargs, layer_type))
    
    return layer_list


