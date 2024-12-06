from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from zarr.abc.codec import (
    Codec,
    CodecPipeline,
)
from zarr.core.config import config

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import ChunkCoords
    from zarr.core.indexing import SelectorTuple

from ._internal import CodecPipelineImpl
from .utils import (
    CollapsedDimensionError,
    DiscontiguousArrayError,
    make_chunk_info_for_rust,
    make_chunk_info_for_rust_with_indices,
)


def get_codec_pipeline_impl(codec_metadata_json: str) -> CodecPipelineImpl:
    return CodecPipelineImpl(
        codec_metadata_json,
        validate_checksums=config.get("codec_pipeline.validate_checksums", None),
        # TODO: upstream zarr-python array.write_empty_chunks is not merged yet #2429
        store_empty_chunks=config.get("array.write_empty_chunks", None),
        chunk_concurrent_minimum=config.get(
            "codec_pipeline.chunk_concurrent_minimum", None
        ),
        chunk_concurrent_maximum=config.get(
            "codec_pipeline.chunk_concurrent_maximum", None
        ),
        num_threads=config.get("threading.max_workers", None),
    )


class ZarrsCodecPipelineState(TypedDict):
    codec_metadata_json: str
    codecs: tuple[Codec, ...]


@dataclass
class ZarrsCodecPipeline(CodecPipeline):
    codecs: tuple[Codec, ...]
    impl: CodecPipelineImpl
    codec_metadata_json: str

    def __getstate__(self) -> ZarrsCodecPipelineState:
        return {"codec_metadata_json": self.codec_metadata_json, "codecs": self.codecs}

    def __setstate__(self, state: ZarrsCodecPipelineState):
        self.codecs = state["codecs"]
        self.codec_metadata_json = state["codec_metadata_json"]
        self.impl = get_codec_pipeline_impl(self.codec_metadata_json)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        raise NotImplementedError("evolve_from_array_spec")

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec]) -> Self:
        codec_metadata = [codec.to_dict() for codec in codecs]
        codec_metadata_json = json.dumps(codec_metadata)
        # TODO: upstream zarr-python has not settled on how to deal with configs yet
        # Should they be checked when an array is created, or when an operation is performed?
        # https://github.com/zarr-developers/zarr-python/issues/2409
        # https://github.com/zarr-developers/zarr-python/pull/2429
        return cls(
            codec_metadata_json=codec_metadata_json,
            codecs=tuple(codecs),
            impl=get_codec_pipeline_impl(codec_metadata_json),
        )

    @property
    def supports_partial_decode(self) -> bool:
        return False

    @property
    def supports_partial_encode(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Codec]:
        yield from self.codecs

    def validate(
        self, *, shape: ChunkCoords, dtype: np.dtype[Any], chunk_grid: ChunkGrid
    ) -> None:
        raise NotImplementedError("validate")

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        raise NotImplementedError("compute_encoded_size")

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        raise NotImplementedError("decode")

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        raise NotImplementedError("encode")

    async def read(
        self,
        batch_info: Iterable[
            tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),  # FIXME: unused
    ) -> None:
        out = out.as_ndarray_like()  # FIXME: Error if array is not in host memory
        if not out.dtype.isnative:
            raise RuntimeError("Non-native byte order not supported")
        try:
            chunks_desc = make_chunk_info_for_rust_with_indices(batch_info, drop_axes)
            index_in_rust = True
        except (DiscontiguousArrayError, CollapsedDimensionError):
            chunks_desc = make_chunk_info_for_rust(batch_info)
            index_in_rust = False
        if index_in_rust:
            await asyncio.to_thread(
                self.impl.retrieve_chunks_and_apply_index,
                chunks_desc,
                out,
            )
            return None
        chunks = await asyncio.to_thread(self.impl.retrieve_chunks, chunks_desc)
        for chunk, chunk_info in zip(chunks, batch_info):
            out_selection = chunk_info[3]
            selection = chunk_info[2]
            spec = chunk_info[1]
            chunk_reshaped = chunk.view(spec.dtype).reshape(spec.shape)
            chunk_selected = chunk_reshaped[selection]
            if drop_axes:
                chunk_selected = np.squeeze(chunk_selected, axis=drop_axes)
            out[out_selection] = chunk_selected

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple]
        ],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        value = value.as_ndarray_like()  # FIXME: Error if array is not in host memory
        if not value.dtype.isnative:
            value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("="))
        elif not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        chunks_desc = make_chunk_info_for_rust_with_indices(batch_info, drop_axes)
        await asyncio.to_thread(
            self.impl.store_chunks_with_indices,
            chunks_desc,
            value,
        )
        return None
