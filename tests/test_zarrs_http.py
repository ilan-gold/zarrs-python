#!/usr/bin/env python3

import numpy as np
import zarr

import zarrs  # noqa: F401

arr_ref = np.array(
    [
        [np.nan, np.nan, np.nan, np.nan, 0.1, 0.1, -0.6, 0.1],
        [np.nan, np.nan, np.nan, np.nan, 0.1, 0.1, -1.6, 0.1],
        [np.nan, np.nan, np.nan, np.nan, 0.1, 0.1, -2.6, 0.1],
        [np.nan, np.nan, np.nan, np.nan, -3.4, -3.5, -3.6, 0.1],
        [1.0, 1.0, 1.0, -4.3, -4.4, -4.5, -4.6, 1.1],
        [1.0, 1.0, 1.0, -5.3, -5.4, -5.5, -5.6, 1.1],
        [1.0, 1.0, 1.0, 1.0, 1.1, 1.1, -6.6, 1.1],
        [1.0, 1.0, 1.0, 1.0, -7.4, -7.5, -7.6, -7.7],
    ]
)


def test_zarrs_http():
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    arr = zarr.open(
        "https://raw.githubusercontent.com/LDeakin/zarrs/main/zarrs/tests/data/array_write_read.zarr/group/array"
    )
    assert arr.shape == (8, 8)
    assert np.allclose(arr[:], arr_ref, equal_nan=True)
