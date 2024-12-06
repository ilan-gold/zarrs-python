#!/usr/bin/env python3

import operator
import pickle
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from functools import reduce
from itertools import product
from pathlib import Path
from types import EllipsisType

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import zarrs  # noqa: F401

axis_size_ = 10
chunk_size_ = axis_size_ // 2
fill_value_ = 32767
dimensionalities_ = list(range(1, 5))


@pytest.fixture
def fill_value() -> int:
    return fill_value_


non_numpy_indices = [
    pytest.param(slice(1, 3), id="slice_in_chunk"),
    pytest.param(slice(1, 7), id="slice_across_chunks"),
    pytest.param(2, id="int"),
    pytest.param(slice(None), id="full_slice"),
    pytest.param(Ellipsis, id="ellipsis"),
]

numpy_indices = [
    pytest.param(np.array([1, 2]), id="contiguous_in_chunk_array"),
    pytest.param(np.array([0, 3]), id="discontinuous_in_chunk_array"),
    pytest.param(np.array([0, 6]), id="across_chunks_indices_array"),
]

all_indices = numpy_indices + non_numpy_indices

indexing_method_params = [
    pytest.param(lambda x: getattr(x, "oindex"), id="oindex"),
    pytest.param(lambda x: x, id="vindex"),
]


def pytest_generate_tests(metafunc):
    old_pipeline_path = zarr.config.get("codec_pipeline.path")
    # need to set the codec pipeline to the zarrs pipeline because the autouse fixture doesn't apply here
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    if "test_roundtrip" in metafunc.function.__name__:
        arrs = []
        indices = []
        store_values = []
        indexing_methods = []
        ids = []
        for dimensionality in dimensionalities_:
            indexers = non_numpy_indices if dimensionality > 2 else all_indices
            for index_param_prod in product(indexers, repeat=dimensionality):
                index = tuple(index_param.values[0] for index_param in index_param_prod)
                # multi-ellipsis indexing is not supported
                if sum(isinstance(i, EllipsisType) for i in index) > 1:
                    continue
                for indexing_method_param in indexing_method_params:
                    arr = gen_arr(fill_value_, Path(tempfile.mktemp()), dimensionality)
                    indexing_method = indexing_method_param.values[0]
                    dimensionality_id = f"{dimensionality}d"
                    id = "-".join(
                        [indexing_method_param.id, dimensionality_id]
                        + [index_param.id for index_param in index_param_prod]
                    )
                    ids.append(id)
                    store_values.append(
                        gen_store_values(
                            indexing_method,
                            index,
                            full_array((axis_size_,) * dimensionality),
                        )
                    )
                    indexing_methods.append(indexing_method)
                    indices.append(index)
                    arrs.append(arr)
        # array is used as param name to prevent collision with arr fixture
        metafunc.parametrize(
            ["array", "index", "store_values", "indexing_method"],
            zip(arrs, indices, store_values, indexing_methods),
            ids=ids,
        )
        zarr.config.set({"codec_pipeline.path": old_pipeline_path})


def full_array(shape) -> np.ndarray:
    return np.arange(reduce(operator.mul, shape, 1)).reshape(shape)


def gen_store_values(
    indexing_method: Callable,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    full_array: np.ndarray,
) -> np.ndarray:
    class smoke:
        oindex = "oindex"

    def maybe_convert(
        i: int | np.ndarray | slice | EllipsisType, axis: int
    ) -> np.ndarray:
        if isinstance(i, np.ndarray):
            return i
        if isinstance(i, slice):
            return np.arange(
                i.start if i.start is not None else 0,
                i.stop if i.stop is not None else full_array.shape[axis],
            )
        if isinstance(i, int):
            return np.array([i])
        if isinstance(i, EllipsisType):
            return np.arange(full_array.shape[axis])
        raise ValueError(f"Invalid index {i}")

    if not isinstance(index, EllipsisType) and indexing_method(smoke()) == "oindex":
        index: tuple[np.ndarray, ...] = tuple(
            maybe_convert(i, axis) for axis, i in enumerate(index)
        )
        res = full_array[np.ix_(*index)]
        # squeeze out extra dims from integer indexers
        if all(i.shape == (1,) for i in index):
            res = res.squeeze()
            return res
        res = res.squeeze(
            axis=tuple(axis for axis, i in enumerate(index) if i.shape == (1,))
        )
        return res
    return full_array[index]


def gen_arr(fill_value, tmp_path, dimensionality) -> zarr.Array:
    return zarr.create(
        (axis_size_,) * dimensionality,
        store=LocalStore(root=tmp_path / ".zarr"),
        chunks=(chunk_size_,) * dimensionality,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )


@pytest.fixture(params=dimensionalities_)
def dimensionality(request):
    return request.param


@pytest.fixture
def arr(dimensionality, tmp_path) -> zarr.Array:
    return gen_arr(fill_value_, tmp_path, dimensionality)


def test_fill_value(arr: zarr.Array):
    assert np.all(arr[:] == fill_value_)


def test_constant(arr: zarr.Array):
    arr[:] = 42
    assert np.all(arr[:] == 42)


def test_singleton(arr: zarr.Array):
    singleton_index = (1,) * len(arr.shape)
    non_singleton_index = (0,) * len(arr.shape)
    arr[singleton_index] = 42
    assert arr[singleton_index] == 42
    assert arr[non_singleton_index] != 42


def test_full_array(arr: zarr.Array):
    stored_values = full_array(arr.shape)
    arr[:] = stored_values
    assert np.all(arr[:] == stored_values)


def test_roundtrip(
    array: zarr.Array,
    store_values: np.ndarray,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    indexing_method(array)[index] = store_values
    res = indexing_method(array)[index]
    assert np.all(
        res == store_values,
    ), res


@contextmanager
def use_zarr_default_codec_reader():
    zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    )
    yield
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def test_roundtrip_read_only_zarrs(
    array: zarr.Array,
    store_values: np.ndarray,
    index: tuple[int | slice | np.ndarray | EllipsisType, ...],
    indexing_method: Callable,
):
    with use_zarr_default_codec_reader():
        arr_default = zarr.open(array.store, read_only=True)
        indexing_method(arr_default)[index] = store_values
    res = indexing_method(zarr.open(array.store))[index]
    assert np.all(
        res == store_values,
    ), res


def test_ellipsis_indexing_invalid(arr: zarr.Array):
    if len(arr.shape) <= 2:
        pytest.skip(
            "Ellipsis indexing works for 1D and 2D arrays in zarr-python despite a shape mismatch"
        )
    stored_value = np.array([1, 2, 3])
    with pytest.raises(ValueError):  # noqa: PT011
        # zarrs-python error: ValueError: operands could not be broadcast together with shapes (4,) (3,)
        # numpy error: ValueError: could not broadcast input array from shape (3,) into shape (4,)
        arr[2, ...] = stored_value


def test_pickle(arr: zarr.Array, tmp_path: Path):
    arr[:] = np.arange(reduce(operator.mul, arr.shape, 1)).reshape(arr.shape)
    expected = arr[:]
    with Path.open(tmp_path / "arr.pickle", "wb") as f:
        pickle.dump(arr._async_array.codec_pipeline, f)
    with Path.open(tmp_path / "arr.pickle", "rb") as f:
        object.__setattr__(arr._async_array, "codec_pipeline", pickle.load(f))
    assert (arr[:] == expected).all()
