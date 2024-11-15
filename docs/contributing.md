# Contributing

## Rust

You will need `rust` and `cargo` installed on your local system.  For more info, see [the rust docs](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Environment management

We encourage the use of [uv](https://docs.astral.sh/uv/) for environment management.  To install the package for development, run

```
uv pip install -e ".[test,dev,doc]"
```

However, take note that while this does build the rust package, the rust package will not be rebuilt upon edits despite the `-e` flag.  You will need to manually rebuild it using either `uv pip install -e .` or `maturin develop`.  Take note that for benchmarking/speed testing, it is advisable to build a release version of the rust package by passing the `-r` flag to `maturin`.  For more information on the `rust`-`python` bridge, see the [`PyO3` docs](https://pyo3.rs/v0.22.6/).

## Testing

To install test dependencies, simply run 

```
pytest
```

or

```
pytest -n auto
```

for parallelized tests.  Most tests have been copied from the `zarr-python` repository with the exception of `test_pipeline.py` which we have written.