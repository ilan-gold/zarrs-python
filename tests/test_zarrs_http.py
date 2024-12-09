#!/usr/bin/env python3

import aiohttp
import numpy as np
import pytest
import zarr
from zarr.storage.remote import RemoteStore

ARR_REF = np.array(
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

URL = "https://raw.githubusercontent.com/LDeakin/zarrs/main/zarrs/tests/data/array_write_read.zarr/group/array"


def test_zarrs_http():
    arr = zarr.open(URL)
    assert arr.shape == (8, 8)
    assert np.allclose(arr[:], ARR_REF, equal_nan=True)


@pytest.mark.xfail(reason="Storage options are not supported for HTTP store")
def test_zarrs_http_kwargs():
    store = RemoteStore.from_url(
        URL, storage_options={"auth": aiohttp.BasicAuth("user", "pass")}
    )
    arr = zarr.open(store)
    assert arr.shape == (8, 8)
    assert np.allclose(arr[:], ARR_REF, equal_nan=True)
