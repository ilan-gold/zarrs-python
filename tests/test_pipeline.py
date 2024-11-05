#!/usr/bin/env python3

import tempfile

import numpy as np
import pytest
import zarr
import zarrs_python  # noqa: F401
from zarr.storage import LocalStore


@pytest.fixture
def fill_value() -> int:
    return 32767


@pytest.fixture
def chunks() -> tuple[int, ...]:
    return (2, 2)


@pytest.fixture(params=[np.array([1, 2]), slice(1, 3)], ids=["array", "slice"])
def indexer(request) -> slice | np.ndarray:
    return request.param


indexer_2 = indexer


@pytest.fixture
def arr(fill_value, chunks) -> zarr.Array:
    shape = (4, 4)

    tmp = tempfile.TemporaryDirectory()
    return zarr.create(
        shape,
        store=LocalStore(root=tmp.name, mode="w"),
        chunks=chunks,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )


def test_fill_value(arr: zarr.Array, fill_value: int):
    assert np.all(arr[:] == fill_value)


def test_store_constant(arr: zarr.Array):
    arr[:] = 42
    assert np.all(arr[:] == 42)


def test_store_singleton(arr: zarr.Array):
    arr[1, 1] = 42
    assert arr[1, 1] == 42
    assert arr[0, 0] != 42


def test_decode(arr: zarr.Array, chunks):
    stored_values = np.arange(16).reshape(4, 4)
    arr[:] = stored_values
    assert np.all(arr[:] == stored_values)
    assert np.all(
        arr[tuple(slice(0, chunk) for chunk in chunks)] == np.array([[0, 1], [4, 5]])
    )
    assert np.all(arr[0:1, 1:2] == np.array([[1]]))
    assert np.all(arr[1:3, 1:3] == np.array([[5, 6], [9, 10]]))


def test_encode_partial(
    arr: zarr.Array,
    fill_value: int,
    indexer: slice | np.ndarray,
    indexer_2: slice | np.ndarray,
):
    if isinstance(indexer, np.ndarray) and isinstance(indexer_2, np.ndarray):
        pytest.skip(
            "indexing across two axes with arrays seems to have strange behavior even in normal zarr"
        )
    stored_value = np.array([[-1, -2], [-3, -4]])
    arr[indexer, indexer_2] = stored_value
    res = arr[:]
    assert np.all(
        res
        == np.array(
            [
                [fill_value] * arr.shape[0],
                [fill_value, -1, -2, fill_value],
                [fill_value, -3, -4, fill_value],
                [fill_value] * arr.shape[0],
            ]
        ),
    ), res
    res = arr[indexer, indexer_2]
    assert np.all(
        res == stored_value,
    ), res


def test_encode_singleton_axis(
    arr: zarr.Array, fill_value: int, indexer: slice | np.ndarray
):
    stored_value = np.array([-3, -4])
    arr[2, indexer] = stored_value
    res = arr[:]
    assert np.all(
        res
        == np.array(
            [
                [fill_value] * arr.shape[0],
                [fill_value] * arr.shape[0],
                [fill_value, -3, -4, fill_value],
                [fill_value] * arr.shape[0],
            ]
        ),
    ), res
    res = arr[2, indexer]
    assert np.all(res == stored_value), res
