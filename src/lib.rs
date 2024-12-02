#![warn(clippy::pedantic)]

use chunk_item::{ChunksItem, IntoItem};
use concurrency::ChunkConcurrentLimitAndCodecOptions;
use numpy::npyffi::PyArrayObject;
use numpy::{IntoPyArray, PyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, CodecOptions, CodecOptionsBuilder, StoragePartialDecoder,
};
use zarrs::array::{
    copy_fill_value_into, update_array_bytes, ArrayBytes, ArraySize, CodecChain, FillValue,
};
use zarrs::array_subset::ArraySubset;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::{ReadableWritableListableStorageTraits, StorageHandle, StoreKey};

mod chunk_item;
mod codec_pipeline_store_filesystem;
mod codec_pipeline_store_http;
mod concurrency;
mod runtime;
#[cfg(test)]
mod tests;
mod utils;

use codec_pipeline_store_filesystem::CodecPipelineStoreFilesystem;
use codec_pipeline_store_http::CodecPipelineStoreHTTP;
use utils::{PyErrExt, PyUntypedArrayExt};

trait CodecPipelineStore: Send + Sync {
    fn store(&self) -> Arc<dyn ReadableWritableListableStorageTraits>;
    fn chunk_path(&self, store_path: &str) -> PyResult<String>;
}

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub struct CodecPipelineImpl {
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) store: Mutex<Option<Arc<dyn CodecPipelineStore>>>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
}

/// A store root and path pair.
// TODO: Prefer a struct with named fields, but I couldn't be bothered to figure out how to do that with stubgen etc
type StorePath = (String, String);

impl CodecPipelineImpl {
    fn get_store_and_path(
        &self,
        store_path: &StorePath,
    ) -> PyResult<(Arc<dyn ReadableWritableListableStorageTraits>, String)> {
        let mut gstore = self.store.lock().map_err(|_| {
            PyErr::new::<PyRuntimeError, _>("failed to lock the store mutex".to_string())
        })?;

        // TODO: Request upstream change to get store on codec pipeline initialisation, do not want to do all of this here
        #[allow(clippy::collapsible_if)]
        if gstore.is_none() {
            if store_path.0.is_empty() && store_path.1.starts_with("file://") {
                *gstore = Some(Arc::new(CodecPipelineStoreFilesystem::new()?));
            } else if store_path.0.starts_with("http://") || store_path.0.starts_with("https://") {
                *gstore = Some(Arc::new(CodecPipelineStoreHTTP::new(&store_path.0)?));
            }
            // TODO: Add support for more stores
        }

        if let Some(gstore) = gstore.as_ref() {
            Ok((gstore.store(), gstore.chunk_path(&store_path.1)?))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(format!(
                "unsupported store for root:{} path:{}",
                store_path.0, store_path.1
            )))
        }
    }

    fn collect_chunk_descriptions<R: IntoItem<I, S>, I, S: Copy>(
        &self,
        chunk_descriptions: Vec<R>,
        shape: S,
    ) -> PyResult<Vec<I>> {
        chunk_descriptions
            .into_iter()
            .map(|raw| {
                let (store, path) = self.get_store_and_path(raw.store_path())?;
                let key = StoreKey::new(path).map_py_err::<PyValueError>()?;
                raw.into_item(store, key, shape)
            })
            .collect()
    }

    fn retrieve_chunk_bytes<'a, I: ChunksItem>(
        item: &I,
        codec_chain: &CodecChain,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = item.get().map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(value_encoded.into(), item.representation(), codec_options)
                .map_py_err::<PyRuntimeError>()?
        } else {
            let array_size = ArraySize::new(
                item.representation().data_type().size(),
                item.representation().num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, item.representation().fill_value())
        };
        Ok(value_decoded)
    }

    fn store_chunk_bytes<I: ChunksItem>(
        item: &I,
        codec_chain: &CodecChain,
        value_decoded: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        value_decoded
            .validate(
                item.representation().num_elements(),
                item.representation().data_type().size(),
            )
            .map_py_err::<PyValueError>()?;

        if value_decoded.is_fill_value(item.representation().fill_value()) {
            item.store().erase(item.key())
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, item.representation(), codec_options)
                .map(Cow::into_owned)
                .map_py_err::<PyRuntimeError>()?;

            // Store the encoded chunk
            item.store().set(item.key(), value_encoded.into())
        }
        .map_py_err::<PyRuntimeError>()
    }

    fn store_chunk_subset_bytes<I: ChunksItem>(
        item: &I,
        codec_chain: &CodecChain,
        chunk_subset_bytes: ArrayBytes,
        chunk_subset: &ArraySubset,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        if !chunk_subset.inbounds(&item.representation().shape_u64()) {
            return Err(PyErr::new::<PyValueError, _>(
                "chunk subset is out of bounds".to_string(),
            ));
        }

        if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == item.representation().shape_u64()
        {
            // Fast path if the chunk subset spans the entire chunk, no read required
            Self::store_chunk_bytes(item, codec_chain, chunk_subset_bytes, codec_options)
        } else {
            // Validate the chunk subset bytes
            chunk_subset_bytes
                .validate(
                    chunk_subset.num_elements(),
                    item.representation().data_type().size(),
                )
                .map_py_err::<PyValueError>()?;

            // Retrieve the chunk
            let chunk_bytes_old = Self::retrieve_chunk_bytes(item, codec_chain, codec_options)?;

            // Update the chunk
            let chunk_bytes_new = unsafe {
                // SAFETY:
                // - chunk_bytes_old is compatible with the chunk shape and data type size (validated on decoding)
                // - chunk_subset is compatible with chunk_subset_bytes and the data type size (validated above)
                // - chunk_subset is within the bounds of the chunk shape (validated above)
                // - output bytes and output subset bytes are compatible (same data type)
                update_array_bytes(
                    chunk_bytes_old,
                    &item.representation().shape_u64(),
                    chunk_subset,
                    &chunk_subset_bytes,
                    item.representation().data_type().size(),
                )
            };

            // Store the updated chunk
            Self::store_chunk_bytes(item, codec_chain, chunk_bytes_new, codec_options)
        }
    }

    fn pyarray_itemsize(value: &Bound<'_, PyUntypedArray>) -> usize {
        // TODO: is this and the below a bug? why doesn't .itemsize() work?
        value
            .dtype()
            .getattr("itemsize")
            .unwrap()
            .extract::<usize>()
            .unwrap()
    }

    fn py_untyped_array_to_array_object<'a>(
        value: &Bound<'a, PyUntypedArray>,
    ) -> &'a PyArrayObject {
        let array_object_ptr: *mut PyArrayObject = value.as_array_ptr();
        unsafe {
            // SAFETY: array_object_ptr cannot be null
            &*array_object_ptr
        }
    }

    fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> &'a [u8] {
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * Self::pyarray_itemsize(value);
        let slice = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts(array_data, array_len)
        };
        slice
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> UnsafeCellSlice<'a, u8> {
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * Self::pyarray_itemsize(value);
        let output = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        UnsafeCellSlice::new(output)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CodecPipelineImpl {
    #[pyo3(signature = (
        metadata,
        *,
        validate_checksums=None,
        store_empty_chunks=None,
        chunk_concurrent_minimum=None,
        chunk_concurrent_maximum=None,
        num_threads=None,
    ))]
    #[new]
    fn new(
        metadata: &str,
        validate_checksums: Option<bool>,
        store_empty_chunks: Option<bool>,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let metadata: Vec<MetadataV3> =
            serde_json::from_str(metadata).map_py_err::<PyTypeError>()?;
        let codec_chain =
            Arc::new(CodecChain::from_metadata(&metadata).map_py_err::<PyTypeError>()?);
        let mut codec_options = CodecOptionsBuilder::new();
        if let Some(validate_checksums) = validate_checksums {
            codec_options = codec_options.validate_checksums(validate_checksums);
        }
        if let Some(store_empty_chunks) = store_empty_chunks {
            codec_options = codec_options.store_empty_chunks(store_empty_chunks);
        }
        let codec_options = codec_options.build();

        let chunk_concurrent_minimum = chunk_concurrent_minimum
            .unwrap_or(zarrs::config::global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        Ok(Self {
            codec_chain,
            store: Mutex::new(None),
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
        })
    }

    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::RawWithIndices>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let output = Self::nparray_to_unsafe_cell_slice(value);
        let output_shape: Vec<u64> = value.shape_zarr()?;
        let chunk_descriptions =
            self.collect_chunk_descriptions(chunk_descriptions, &output_shape)?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        py.allow_threads(move || {
            let update_chunk_subset = |item: chunk_item::WithSubset| {
                // See zarrs::array::Array::retrieve_chunk_subset_into
                if item.chunk_subset.start().iter().all(|&o| o == 0)
                    && item.chunk_subset.shape() == item.representation().shape_u64()
                {
                    // See zarrs::array::Array::retrieve_chunk_into
                    let chunk_encoded = item.get().map_py_err::<PyRuntimeError>()?;
                    if let Some(chunk_encoded) = chunk_encoded {
                        // Decode the encoded data into the output buffer
                        let chunk_encoded: Vec<u8> = chunk_encoded.into();
                        unsafe {
                            // SAFETY:
                            // - output is an array with output_shape elements of the item.representation data type,
                            // - item.subset is within the bounds of output_shape.
                            self.codec_chain.decode_into(
                                Cow::Owned(chunk_encoded),
                                item.representation(),
                                &output,
                                &output_shape,
                                &item.subset,
                                &codec_options,
                            )
                        }
                    } else {
                        // The chunk is missing, write the fill value
                        unsafe {
                            // SAFETY:
                            // - data type and fill value are confirmed to be compatible when the ChunkRepresentation is created,
                            // - output is an array with output_shape elements of the item.representation data type,
                            // - item.subset is within the bounds of output_shape.
                            copy_fill_value_into(
                                item.representation().data_type(),
                                item.representation().fill_value(),
                                &output,
                                &output_shape,
                                &item.subset,
                            )
                        }
                    }
                } else {
                    // Partially decode the chunk into the output buffer
                    let storage_handle = Arc::new(StorageHandle::new(item.store().clone()));
                    // NOTE: Normally a storage transformer would exist between the storage handle and the input handle
                    // but zarr-python does not support them nor forward them to the codec pipeline
                    let input_handle = Arc::new(StoragePartialDecoder::new(
                        storage_handle,
                        item.key().clone(),
                    ));
                    let partial_decoder = self
                        .codec_chain
                        .clone()
                        .partial_decoder(input_handle, item.representation(), &codec_options)
                        .map_py_err::<PyValueError>()?;
                    unsafe {
                        // SAFETY:
                        // - output is an array with output_shape elements of the item.representation data type,
                        // - item.subset is within the bounds of output_shape.
                        // - item.chunk_subset has the same number of elements as item.subset.
                        partial_decoder.partial_decode_into(
                            &item.chunk_subset,
                            &output,
                            &output_shape,
                            &item.subset,
                            &codec_options,
                        )
                    }
                }
                .map_py_err::<PyValueError>()
            };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                update_chunk_subset
            )?;

            Ok(())
        })
    }

    fn retrieve_chunks<'py>(
        &self,
        py: Python<'py>,
        chunk_descriptions: Vec<chunk_item::Raw>, // FIXME: Ref / iterable?
    ) -> PyResult<Vec<Bound<'py, PyArray1<u8>>>> {
        let chunk_descriptions = self.collect_chunk_descriptions(chunk_descriptions, ())?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(vec![]);
        };

        let chunk_bytes = py.allow_threads(move || {
            let get_chunk_subset = |item: chunk_item::Basic| {
                let chunk_encoded = item.get().map_py_err::<PyRuntimeError>()?;
                Ok(if let Some(chunk_encoded) = chunk_encoded {
                    let chunk_encoded: Vec<u8> = chunk_encoded.into();
                    self.codec_chain
                        .decode(
                            Cow::Owned(chunk_encoded),
                            item.representation(),
                            &codec_options,
                        )
                        .map_py_err::<PyRuntimeError>()?
                } else {
                    // The chunk is missing so we need to create one.
                    let num_elements = item.representation().num_elements();
                    let data_type_size = item.representation().data_type().size();
                    let chunk_shape = ArraySize::new(data_type_size, num_elements);
                    ArrayBytes::new_fill_value(chunk_shape, item.representation().fill_value())
                }
                .into_fixed()
                .map_py_err::<PyRuntimeError>()?
                .into_owned())
            };
            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                map,
                get_chunk_subset
            )
            .collect::<PyResult<Vec<Vec<u8>>>>()
        })?;
        Ok(chunk_bytes
            .into_iter()
            .map(|x| x.into_pyarray(py))
            .collect())
    }

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::RawWithIndices>,
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        enum InputValue<'a> {
            Array(ArrayBytes<'a>),
            Constant(FillValue),
        }

        // Get input array
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }

        let input_slice = Self::nparray_to_slice(value);
        let input = if value.ndim() > 0 {
            InputValue::Array(ArrayBytes::new_flen(Cow::Borrowed(input_slice)))
        } else {
            InputValue::Constant(FillValue::new(input_slice.to_vec()))
        };

        let input_shape: Vec<u64> = value.shape_zarr()?;
        let chunk_descriptions =
            self.collect_chunk_descriptions(chunk_descriptions, &input_shape)?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        py.allow_threads(move || {
            let store_chunk = |item: chunk_item::WithSubset| match &input {
                InputValue::Array(input) => {
                    let chunk_subset_bytes = input
                        .extract_array_subset(
                            &item.subset,
                            &input_shape,
                            item.item.representation().data_type(),
                        )
                        .map_py_err::<PyRuntimeError>()?;
                    Self::store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &item.chunk_subset,
                        &codec_options,
                    )
                }
                InputValue::Constant(constant_value) => {
                    let chunk_subset_bytes = ArrayBytes::new_fill_value(
                        ArraySize::new(
                            item.representation().data_type().size(),
                            item.chunk_subset.num_elements(),
                        ),
                        constant_value,
                    );

                    Self::store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &item.chunk_subset,
                        &codec_options,
                    )
                }
            };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                store_chunk
            )?;

            Ok(())
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
