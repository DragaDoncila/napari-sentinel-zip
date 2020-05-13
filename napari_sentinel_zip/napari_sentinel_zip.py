"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import numpy as np
import re
from napari_plugin_engine import napari_hook_implementation
from glob import glob

SENTINEL_PATH_REGEX = re.compile(r"SENTINEL.*_[0-9]{8}.*\.zip")


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
    # if we've been handed a single zip that's fine
    if isinstance(path, str) and SENTINEL_PATH_REGEX.match(path):
        return reader_function

    # if we've been hnaded a list of SENTINEL zips, that's fine
    if isinstance(path, list):
        # all paths within must be sentinel zips
        for pth in path:
            if not SENTINEL_PATH_REGEX.match(pth):
                return None
    
    # if we've been handed a root directory with SENTINEL zips inside, that's fine
    all_zips = glob(path + '/*.zip')
    filtered_zips = filter(SENTINEL_PATH_REGEX.match, all_zips)
    if not len(all_zips):
        return None
    path = all_zips

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
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    # https://napari.org/docs/api/napari.components.html#module-napari.components.add_layers_mixin
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
