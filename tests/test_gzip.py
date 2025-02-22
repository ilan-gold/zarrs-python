import numpy as np
from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import BytesCodec, GzipCodec
from zarr.storage import StorePath


def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        StorePath(store),
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[BytesCodec(), GzipCodec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
