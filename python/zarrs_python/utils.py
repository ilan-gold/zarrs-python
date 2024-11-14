from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.indexing import SelectorTuple, is_integer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import EllipsisType

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import ChunkCoords


# adapted from https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
def get_max_threads() -> int:
    return (os.cpu_count() or 1) + 4


class DiscontiguousArrayError(Exception):
    pass


class CollapsedDimensionError(Exception):
    pass


# This is a (mostly) copy of the function from zarr.core.indexing that fixes:
#   DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
# TODO: Upstream this fix
def make_slice_selection(selection: tuple[np.ndarray | float]) -> list[slice]:
    ls: list[slice] = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            dim_selection = dim_selection.ravel()
            if len(dim_selection) == 1:
                ls.append(
                    slice(int(dim_selection.item()), int(dim_selection.item()) + 1, 1)
                )
            else:
                diff = np.diff(dim_selection)
                if (diff != 1).any() and (diff != 0).any():
                    raise DiscontiguousArrayError(diff)
                ls.append(slice(dim_selection[0], dim_selection[-1] + 1, 1))
        else:
            ls.append(dim_selection)
    if (
        sum(isinstance(dim_selection, np.ndarray) for dim_selection in selection)
        == sum(isinstance(dim_selection, slice) for dim_selection in selection)
        == len(selection)
    ):
        raise DiscontiguousArrayError(
            "Vindexing with only contiguous numpy arrays is not supported"
        )
    return ls


def selector_tuple_to_slice_selection(selector_tuple: SelectorTuple) -> list[slice]:
    if isinstance(selector_tuple, slice):
        return [selector_tuple]
    if all(isinstance(s, slice) for s in selector_tuple):
        return list(selector_tuple)
    return make_slice_selection(selector_tuple)


def convert_chunk_to_primitive(
    byte_getter: ByteGetter | ByteSetter, chunk_spec: ArraySpec
) -> tuple[str, ChunkCoords, str, Any]:
    return (
        str(byte_getter),
        chunk_spec.shape,
        str(chunk_spec.dtype),
        chunk_spec.fill_value.tobytes(),
    )


def resulting_shape_from_index(
    array_shape: tuple[int, ...],
    index_tuple: tuple[int | slice | EllipsisType | np.ndarray],
    drop_axes: tuple[int, ...],
    *,
    pad: bool,
) -> tuple[int, ...]:
    result_shape = []
    advanced_index_shapes = [
        idx.shape for idx in index_tuple if isinstance(idx, np.ndarray)
    ]
    basic_shape_index = 0

    # Broadcast all advanced indices, if any
    if advanced_index_shapes:
        broadcasted_shape = np.broadcast_shapes(*advanced_index_shapes)
        result_shape.extend(broadcasted_shape)
        basic_shape_index += len(
            advanced_index_shapes
        )  # Consume dimensions from array_shape

    # Process each remaining index in index_tuple
    for idx in index_tuple:
        if isinstance(idx, int):
            # Integer index reduces dimension, so skip this dimension in array_shape
            basic_shape_index += 1
        elif isinstance(idx, slice):
            if idx.step is not None and idx.step > 1:
                raise DiscontiguousArrayError(
                    "Step size greater than 1 is not supported"
                )
            # Slice keeps dimension, adjust size accordingly
            start, stop, _ = idx.indices(array_shape[basic_shape_index])
            result_shape.append(stop - start)
            basic_shape_index += 1
        elif idx is Ellipsis:
            # Calculate number of dimensions that Ellipsis should fill
            num_to_fill = len(array_shape) - len(index_tuple) + 1
            result_shape.extend(
                array_shape[basic_shape_index : basic_shape_index + num_to_fill]
            )
            basic_shape_index += num_to_fill

    # Step 4: Append remaining dimensions from array_shape if fewer indices were used
    if basic_shape_index < len(array_shape) and pad:
        result_shape.extend(array_shape[basic_shape_index:])

    return tuple(size for idx, size in enumerate(result_shape) if idx not in drop_axes)


def get_shape_for_selector(
    selector_tuple: SelectorTuple,
    shape: tuple[int, ...],
    drop_axes: tuple[int, ...],
    *,
    pad: bool,
) -> tuple[int, ...]:
    if isinstance(selector_tuple, slice | np.ndarray):
        return resulting_shape_from_index(
            shape,
            (selector_tuple,),
            drop_axes,
            pad=pad,
        )
    return resulting_shape_from_index(shape, selector_tuple, drop_axes, pad=pad)


def make_chunk_info_for_rust_with_indices(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
    ],
    drop_axes: tuple,
) -> list[tuple[tuple[str, ChunkCoords, str, Any], list[slice], list[slice]]]:
    # all?
    for _, chunk_spec, chunk_selection, out_selection in batch_info:
        shape_out_selection = get_shape_for_selector(
            out_selection, chunk_spec.shape, (), pad=False
        )
        shape_chunk_selection = get_shape_for_selector(
            chunk_selection, chunk_spec.shape, drop_axes, pad=True
        )
        if len(shape_chunk_selection) != len(shape_out_selection):
            raise CollapsedDimensionError()
    return list(
        (
            convert_chunk_to_primitive(byte_getter, chunk_spec),
            selector_tuple_to_slice_selection(out_selection),
            selector_tuple_to_slice_selection(chunk_selection),
        )
        for (byte_getter, chunk_spec, chunk_selection, out_selection) in batch_info
    )


def make_chunk_info_for_rust(
    batch_info: Iterable[
        tuple[ByteGetter | ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
    ],
) -> list[tuple[str, ChunkCoords, str, Any]]:
    return list(
        convert_chunk_to_primitive(byte_getter, chunk_spec)
        for (byte_getter, chunk_spec, _, _) in batch_info
    )
