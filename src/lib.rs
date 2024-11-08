#![warn(clippy::pedantic)]

use numpy::npyffi::PyArrayObject;
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{
    ArrayToBytesCodecTraits, CodecOptions, CodecOptionsBuilder, StoragePartialDecoder,
};
use zarrs::array::{
    copy_fill_value_into, update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation,
    CodecChain, DataType, FillValue,
};
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::{ReadableWritableListableStorageTraits, StorageHandle, StoreKey};

mod utils;

use utils::PyErrExt;

pub enum CodecPipelineStore {
    Filesystem(Arc<FilesystemStore>),
}

#[pyclass]
pub struct CodecPipelineImpl {
    pub codec_chain: Arc<CodecChain>,
    pub store: Arc<Mutex<Option<CodecPipelineStore>>>,
    codec_options: CodecOptions,
}

impl CodecPipelineImpl {
    fn get_store_and_path<'a>(
        &self,
        chunk_path: &'a str,
    ) -> PyResult<(Arc<dyn ReadableWritableListableStorageTraits>, &'a str)> {
        let mut gstore = self.store.lock().map_err(|_| {
            PyErr::new::<PyRuntimeError, _>("failed to lock the store mutex".to_string())
        })?;
        if let Some(chunk_path) = chunk_path.strip_prefix("file://") {
            if gstore.is_none() {
                if let Some(chunk_path) = chunk_path.strip_prefix('/') {
                    // Absolute path
                    let store = Arc::new(FilesystemStore::new("/").map_py_err::<PyRuntimeError>()?);
                    *gstore = Some(CodecPipelineStore::Filesystem(store.clone()));
                    Ok((store, chunk_path))
                } else {
                    // Relative path
                    let store = Arc::new(
                        FilesystemStore::new(
                            std::env::current_dir().map_py_err::<PyRuntimeError>()?,
                        )
                        .map_py_err::<PyRuntimeError>()?,
                    );
                    *gstore = Some(CodecPipelineStore::Filesystem(store.clone()));
                    Ok((store, chunk_path))
                }
            } else if let Some(CodecPipelineStore::Filesystem(store)) = gstore.as_ref() {
                if let Some(chunk_path) = chunk_path.strip_prefix('/') {
                    Ok((store.clone(), chunk_path))
                } else {
                    Ok((store.clone(), chunk_path))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "the store type changed".to_string(),
                ))
            }
        } else {
            // TODO: Add support for more stores
            Err(PyErr::new::<PyTypeError, _>(format!(
                "unsupported store for {chunk_path}"
            )))
        }
    }

    fn collect_chunk_descriptions(
        &self,
        chunk_descriptions: Vec<ChunksItemRaw>,
        shape: &[u64],
    ) -> PyResult<Vec<ChunksItem>> {
        chunk_descriptions
            .into_iter()
            .map(
                |(chunk_path, chunk_shape, dtype, fill_value, selection, chunk_selection)| {
                    let (store, path) = self.get_store_and_path(&chunk_path)?;
                    let key = StoreKey::new(path).map_py_err::<PyValueError>()?;
                    Ok(ChunksItem {
                        store,
                        key,
                        chunk_subset: Self::selection_to_array_subset(
                            &chunk_selection,
                            &chunk_shape,
                        )?,
                        subset: Self::selection_to_array_subset(&selection, shape)?,
                        representation: Self::get_chunk_representation(
                            chunk_shape,
                            &dtype,
                            fill_value,
                        )?,
                    })
                },
            )
            .collect()
    }

    fn get_chunk_representation(
        chunk_shape: Vec<u64>,
        dtype: &str,
        fill_value: Vec<u8>,
    ) -> PyResult<ChunkRepresentation> {
        // Get the chunk representation
        let data_type =
            DataType::from_metadata(&DataTypeMetadataV3::from_metadata(&MetadataV3::new(dtype)))
                .map_py_err::<PyRuntimeError>()?;
        let chunk_shape = chunk_shape
            .into_iter()
            .map(|x| NonZeroU64::new(x).expect("chunk shapes should always be non-zero"))
            .collect();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape, data_type, FillValue::new(fill_value))
                .map_py_err::<PyValueError>()?;
        Ok(chunk_representation)
    }

    fn retrieve_chunk_bytes<'a>(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = store.get(key).map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(value_encoded.into(), chunk_representation, codec_options)
                .map_py_err::<PyRuntimeError>()?
        } else {
            let array_size = ArraySize::new(
                chunk_representation.data_type().size(),
                chunk_representation.num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, chunk_representation.fill_value())
        };
        Ok(value_decoded)
    }

    fn store_chunk_bytes(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        value_decoded: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        if value_decoded.is_fill_value(chunk_representation.fill_value()) {
            store.erase(key)
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, chunk_representation, codec_options)
                .map(Cow::into_owned)
                .map_py_err::<PyRuntimeError>()?;

            // Store the encoded chunk
            store.set(key, value_encoded.into())
        }
        .map_py_err::<PyRuntimeError>()
    }

    fn store_chunk_subset_bytes(
        store: &dyn ReadableWritableListableStorageTraits,
        key: &StoreKey,
        codec_chain: &CodecChain,
        chunk_representation: &ChunkRepresentation,
        chunk_subset_bytes: &ArrayBytes,
        chunk_subset: &ArraySubset,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        // Validate the inputs
        chunk_subset_bytes
            .validate(
                chunk_subset.num_elements(),
                chunk_representation.data_type().size(),
            )
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        if !chunk_subset.inbounds(&chunk_representation.shape_u64()) {
            return Err(PyErr::new::<PyValueError, _>(
                "chunk subset is out of bounds".to_string(),
            ));
        }

        // Retrieve the chunk
        let chunk_bytes_old = Self::retrieve_chunk_bytes(
            store,
            key,
            codec_chain,
            chunk_representation,
            codec_options,
        )?;

        // Update the chunk
        let chunk_bytes_new = unsafe {
            // SAFETY:
            // - chunk_bytes_old is compatible with the chunk shape and data type size (validated on decoding)
            // - chunk_subset is compatible with chunk_subset_bytes and the data type size (validated above)
            // - chunk_subset is within the bounds of the chunk shape (validated above)
            // - output bytes and output subset bytes are compatible (same data type)
            update_array_bytes(
                chunk_bytes_old,
                &chunk_representation.shape_u64(),
                chunk_subset,
                chunk_subset_bytes,
                chunk_representation.data_type().size(),
            )
        };

        // Store the updated chunk
        Self::store_chunk_bytes(
            store,
            key,
            codec_chain,
            chunk_representation,
            chunk_bytes_new,
            codec_options,
        )
    }

    fn slice_to_range(slice: &Bound<'_, PySlice>, length: isize) -> PyResult<std::ops::Range<u64>> {
        let indices = slice.indices(length)?;
        if indices.start < 0 {
            Err(PyErr::new::<PyValueError, _>(
                "slice start must be greater than or equal to 0".to_string(),
            ))
        } else if indices.stop < 0 {
            Err(PyErr::new::<PyValueError, _>(
                "slice stop must be greater than or equal to 0".to_string(),
            ))
        } else if indices.step != 1 {
            Err(PyErr::new::<PyValueError, _>(
                "slice step must be equal to 1".to_string(),
            ))
        } else {
            Ok(u64::try_from(indices.start)?..u64::try_from(indices.stop)?)
        }
    }

    fn selection_to_array_subset(
        selection: &[Bound<'_, PySlice>],
        shape: &[u64],
    ) -> PyResult<ArraySubset> {
        if selection.is_empty() {
            Ok(ArraySubset::new_with_shape(vec![1; shape.len()]))
        } else {
            let chunk_ranges = selection
                .iter()
                .zip(shape)
                .map(|(selection, &shape)| Self::slice_to_range(selection, isize::try_from(shape)?))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ArraySubset::new_with_ranges(&chunk_ranges))
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
            // TODO: Verify that empty arrays have non-null data. Otherwise, this function needs to return Option or be unsafe with a non-empty invariant
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
            // TODO: Verify that empty arrays have non-null data. Otherwise, this function needs to return Option or be unsafe with a non-empty invariant
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        UnsafeCellSlice::new(output)
    }
}

type ChunksItemRaw<'a> = (
    // path
    String,
    // shape
    Vec<u64>,
    // data type
    String,
    // fill value bytes
    Vec<u8>,
    // out selection
    Vec<Bound<'a, PySlice>>,
    // chunk selection
    Vec<Bound<'a, PySlice>>,
);

struct ChunksItem {
    store: Arc<dyn ReadableWritableListableStorageTraits>,
    key: StoreKey,
    chunk_subset: ArraySubset,
    subset: ArraySubset,
    representation: ChunkRepresentation,
}

#[pymethods]
impl CodecPipelineImpl {
    #[pyo3(signature = (metadata, validate_checksums=None, store_empty_chunks=None, concurrent_target=None))]
    #[new]
    fn new(
        metadata: &str,
        validate_checksums: Option<bool>,
        store_empty_chunks: Option<bool>,
        concurrent_target: Option<usize>,
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
        if let Some(concurrent_target) = concurrent_target {
            codec_options = codec_options.concurrent_target(concurrent_target);
        }
        let codec_options = codec_options.build();

        Ok(Self {
            codec_chain,
            store: Arc::new(Mutex::new(None)),
            codec_options,
        })
    }

    fn retrieve_chunks(
        &self,
        py: Python,
        chunk_descriptions: Vec<ChunksItemRaw>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
        chunk_concurrent_limit: usize,
    ) -> PyResult<()> {
        // Get input array
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let output = Self::nparray_to_unsafe_cell_slice(value);

        // Get the output shape
        let output_shape: Vec<u64> = if value.shape().is_empty() {
            vec![1] // scalar value
        } else {
            value
                .shape()
                .iter()
                .map(|&i| u64::try_from(i))
                .collect::<Result<_, _>>()?
        };

        let chunk_descriptions =
            self.collect_chunk_descriptions(chunk_descriptions, &output_shape)?;

        py.allow_threads(move || {
            let codec_options = &self.codec_options;

            let update_chunk_subset = |item: ChunksItem| {
                // See zarrs::array::Array::retrieve_chunk_subset_into
                if item.chunk_subset.start().iter().all(|&o| o == 0)
                    && item.chunk_subset.shape() == item.representation.shape_u64()
                {
                    // See zarrs::array::Array::retrieve_chunk_into
                    let chunk_encoded = item.store.get(&item.key).map_py_err::<PyRuntimeError>()?;
                    if let Some(chunk_encoded) = chunk_encoded {
                        // Decode the encoded data into the output buffer
                        let chunk_encoded: Vec<u8> = chunk_encoded.into();
                        unsafe {
                            // SAFETY:
                            // - output is an array with output_shape elements of the item.representation data type,
                            // - item.subset is within the bounds of output_shape.
                            self.codec_chain.decode_into(
                                Cow::Owned(chunk_encoded),
                                &item.representation,
                                &output,
                                &output_shape,
                                &item.subset,
                                codec_options,
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
                                item.representation.data_type(),
                                item.representation.fill_value(),
                                &output,
                                &output_shape,
                                &item.subset,
                            )
                        }
                    }
                } else {
                    // Partially decode the chunk into the output buffer
                    let storage_handle = Arc::new(StorageHandle::new(item.store.clone()));
                    // NOTE: Normally a storage transformer would exist between the storage handle and the input handle
                    // but zarr-python does not support them nor forward them to the codec pipeline
                    let input_handle =
                        Arc::new(StoragePartialDecoder::new(storage_handle, item.key));
                    let partial_decoder = self
                        .codec_chain
                        .clone()
                        .partial_decoder(input_handle, &item.representation, codec_options)
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
                            codec_options,
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

    fn store_chunks(
        &self,
        py: Python,
        chunk_descriptions: Vec<ChunksItemRaw>,
        value: &Bound<'_, PyUntypedArray>,
        chunk_concurrent_limit: usize,
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

        // Get the input shape
        let input_shape: Vec<u64> = if value.shape().is_empty() {
            vec![1] // scalar value
        } else {
            value
                .shape()
                .iter()
                .map(|&i| u64::try_from(i))
                .collect::<Result<_, _>>()?
        };

        let chunk_descriptions =
            self.collect_chunk_descriptions(chunk_descriptions, &input_shape)?;

        py.allow_threads(move || {
            let codec_options = &self.codec_options;

            let store_chunk = |item: ChunksItem| match &input {
                InputValue::Array(input) => {
                    let chunk_subset_bytes = input
                        .extract_array_subset(
                            &item.subset,
                            &input_shape,
                            item.representation.data_type(),
                        )
                        .map_py_err::<PyRuntimeError>()?;
                    Self::store_chunk_subset_bytes(
                        item.store.as_ref(),
                        &item.key,
                        &self.codec_chain,
                        &item.representation,
                        &chunk_subset_bytes,
                        &item.chunk_subset,
                        codec_options,
                    )
                }
                InputValue::Constant(constant_value) => {
                    let chunk_subset_bytes = ArrayBytes::new_fill_value(
                        ArraySize::new(
                            item.representation.data_type().size(),
                            item.chunk_subset.num_elements(),
                        ),
                        constant_value,
                    );

                    Self::store_chunk_subset_bytes(
                        item.store.as_ref(),
                        &item.key,
                        &self.codec_chain,
                        &item.representation,
                        &chunk_subset_bytes,
                        &item.chunk_subset,
                        codec_options,
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
    m.add_class::<CodecPipelineImpl>()?;
    Ok(())
}
