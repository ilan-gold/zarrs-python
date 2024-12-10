# zarrs-python

[![PyPI](https://img.shields.io/pypi/v/zarrs.svg)](https://pypi.org/project/zarrs)
[![Downloads](https://static.pepy.tech/badge/zarrs/month)](https://pepy.tech/project/zarrs)
[![Downloads](https://static.pepy.tech/badge/zarrs)](https://pepy.tech/project/zarrs)
[![Stars](https://img.shields.io/github/stars/ilan-gold/zarrs-python?style=flat&logo=github&color=yellow)](https://github.com/ilan-gold/zarrs-python/stargazers)
![CI](https://github.com/ilan-gold/zarrs-python/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/ilan-gold/zarrs-python/actions/workflows/cd.yml/badge.svg)

```{warning}
⚠️ The version of `zarr-python` we currently depend on is still in pre-release and this
package is accordingly extremely experimental.
We cannot guarantee any stability or correctness at the moment, although we have
tried to do extensive testing and make clear what we think we support and do not.
```

This project serves as a bridge between [`zarrs`](https://docs.rs/zarrs/latest/zarrs/) and [`zarr`](https://zarr.readthedocs.io/en/latest/index.html) via [`PyO3`](https://pyo3.rs/v0.22.3/).  The main goal of the project is to speed up i/o.

To use the project, simply install our package (which depends on `zarr-python>3.0.0b0`), and run:

```python
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

You can then use your `zarr` as normal (with some caveats)!

## API

We export a `ZarrsCodecPipeline` class so that `zarr-python` can use the class but it is not meant to be instantiated and we do not guarantee the stability of its API beyond what is required so that `zarr-python` can use it.  Therefore, it is not documented here.  We also export two errors, `DiscontiguousArrayError` and `CollapsedDimensionError` that can be thrown in the process of converting to indexers that `zarrs` can understand (see below for more details).

At the moment, we only support a subset of the `zarr-python` stores:

- [x] [LocalStore](https://zarr.readthedocs.io/en/main/_autoapi/zarr/storage/local/index.html) (FileSystem)
- [RemoteStore](https://zarr.readthedocs.io/en/main/_autoapi/zarr/storage/remote/index.html)
  - [x] [HTTPFileSystem](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem)

A `NotImplemented` error will be raised if a store is not supported.
We intend to support more stores in the future: https://github.com/ilan-gold/zarrs-python/issues/44.

### Configuration

`ZarrsCodecPipeline` options are exposed through `zarr.config`.

Standard `zarr.config` options control some functionality (see the defaults in the [config.py](https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/config.py) of `zarr-python`):
- `threading.max_workers`: the maximum number of threads used internally by the `ZarrsCodecPipeline` on the Rust side.
  - Defaults to the number of threads in the global `rayon` thread pool if set to `None`, which is [typically the number of logical CPUs](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html#method.num_threads).
- `array.write_empty_chunks`: whether or not to store empty chunks.
  - Defaults to false if `None`. Note that checking for emptiness has some overhead, see [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#store-empty-chunks) for more info.
  - This option name is proposed in [zarr-python #2429](https://github.com/zarr-developers/zarr-python/pull/2429)

The `ZarrsCodecPipeline` specific options are:
- `codec_pipeline.chunk_concurrent_maximum`: the maximum number of chunks stored/retrieved concurrently.
  - Defaults to the number of logical CPUs if `None`. It is constrained by `threading.max_workers` as well.
- `codec_pipeline.chunk_concurrent_minimum`: the minimum number of chunks retrieved/stored concurrently when balancing chunk/codec concurrency.
  - Defaults to 4 if `None`. See [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#chunk-concurrent-minimum) for more info.
- `codec_pipeline.validate_checksums`: enable checksum validation (e.g. with the CRC32C codec).
  - Defaults to true if `None`. See [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#validate-checksums) for more info.

For example:
```python
zarr.config.set({
    "threading.max_workers": None,
    "array.write_empty_chunks": False,
    "codec_pipeline": {
        "path": "zarrs.ZarrsCodecPipeline",
        "validate_checksums": True,
        "store_empty_chunks": False,
        "chunk_concurrent_maximum": None,
        "chunk_concurrent_minimum": 4,
    }
})
```

If the `ZarrsCodecPipeline` is pickled, and then un-pickled, and during that time one of `store_empty_chunks`, `chunk_concurrent_minimum`, `chunk_concurrent_maximum`, or `num_threads` has changed, the newly un-pickled version will pick up the new value.  However, one a `ZarrsCodecPipeline` object has been instantiated, these values are then fixed.  This may change in the future as guidance from the `zarr` community becomes clear.

## Concurrency

Concurrency can be classified into two types:
- chunk (outer) concurrency: the number of chunks retrieved/stored concurrently.
  - This is chosen automatically based on various factors, such as the chunk size and codecs.
  - It is constrained between `codec_pipeline.chunk_concurrent_minimum` and `codec_pipeline.chunk_concurrent_maximum` for operations involving multiple chunks.
- codec (inner) concurrency: the number of threads encoding/decoding a chunk.
  - This is chosen automatically in combination with the chunk concurrency.

The product of the chunk and codec concurrency will approximately match `threading.max_workers`.

Chunk concurrency is typically favored because:
- parallel encoding/decoding can have a high overhead with some codecs, especially with small chunks, and
- it is advantageous to retrieve/store multiple chunks concurrently, especially with high latency stores.

`zarrs-python` will often favor codec concurrency with sharded arrays, as they are well suited to codec concurrency.

## Supported Indexing Methods

We **do not** officially support the following indexing methods.  Some of these methods may error out, others may not:

1. Any `oindex` or `vindex` integer `np.ndarray` indexing with dimensionality >=3 i.e.,

   ```python
   arr[np.array([...]), :, np.array([...])]
   arr[np.array([...]), np.array([...]), np.array([...])]
   arr[np.array([...]), np.array([...]), np.array([...])] = ...
   arr.oindex[np.array([...]), np.array([...]), np.array([...])] = ...
   ```

2. Any `vindex` or `oindex` discontinuous integer `np.ndarray` indexing for writes in 2D

   ```python
   arr[np.array([0, 5]), :] = ...
   arr.oindex[np.array([0, 5]), :] = ...
   ```

3. `vindex` writes in 2D where both indexers are integer `np.ndarray` indices i.e.,

   ```python
   arr[np.array([...]), np.array([...])] = ...
   ```

4. Ellipsis indexing.  We have tested some, but others fail even with `zarr-python`'s default codec pipeline.  Thus for now we advise proceeding with caution here.

   ```python
   arr[0:10, ..., 0:5]
   ```

Otherwise, we believe that we support your indexing case: slices, ints, and all integer `np.ndarray` indices in 2D for reading, contiguous integer `np.ndarray` indices along one axis for writing etc.  Please file an issue if you believe we have more holes in our coverage than we are aware of or you wish to contribute!  For example, we have an [issue in zarrs for integer-array indexing](https://github.com/LDeakin/zarrs/issues/52) that would unblock a lot of these issues!

That being said, using non-contiguous integer `np.ndarray` indexing for reads may not be as fast as expected given the performance of other supported methods.  Until `zarrs` supports integer indexing, only fetching chunks is done in `rust` while indexing then occurs in `python`.
