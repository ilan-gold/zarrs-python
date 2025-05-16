use std::num::NonZeroU64;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyBytes, PyBytesMethods, PyInt, PySlice, PySliceMethods as _},
    Bound, PyAny, PyErr, PyResult,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zarrs::{
    array::{ChunkRepresentation, DataType, FillValue},
    array_subset::ArraySubset,
    metadata::v3::MetadataV3,
    storage::StoreKey,
};

use crate::{store::StoreConfig, utils::PyErrExt};

pub(crate) trait ChunksItem {
    fn store_config(&self) -> StoreConfig;
    fn key(&self) -> &StoreKey;
    fn representation(&self) -> &ChunkRepresentation;
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Basic {
    store: StoreConfig,
    key: StoreKey,
    representation: ChunkRepresentation,
}

fn fill_value_to_bytes(dtype: &str, fill_value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if dtype == "string" {
        // Match zarr-python 2.x.x string fill value behaviour with a 0 fill value
        // See https://github.com/zarr-developers/zarr-python/issues/2792#issuecomment-2644362122
        if let Ok(fill_value_downcast) = fill_value.downcast::<PyInt>() {
            let fill_value_usize: usize = fill_value_downcast.extract()?;
            if fill_value_usize == 0 {
                return Ok(vec![]);
            }
            Err(PyErr::new::<PyValueError, _>(format!(
                    "Cannot understand non-zero integer {fill_value_usize} fill value for dtype {dtype}"
                )))?;
        }
    }

    if let Ok(fill_value_downcast) = fill_value.downcast::<PyBytes>() {
        Ok(fill_value_downcast.as_bytes().to_vec())
    } else if fill_value.hasattr("tobytes")? {
        Ok(fill_value.call_method0("tobytes")?.extract()?)
    } else {
        Err(PyErr::new::<PyValueError, _>(format!(
            "Unsupported fill value {fill_value:?}"
        )))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Basic {
    #[new]
    fn new(byte_interface: &Bound<'_, PyAny>, chunk_spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        let store: StoreConfig = byte_interface.getattr("store")?.extract()?;
        let path: String = byte_interface.getattr("path")?.extract()?;

        let chunk_shape = chunk_spec.getattr("shape")?.extract()?;
        let mut dtype: String = chunk_spec
            .getattr("dtype")?
            .call_method0("__str__")?
            .extract()?;
        if dtype == "object" {
            // zarrs doesn't understand `object` which is the output of `np.dtype("|O").__str__()`
            // but maps it to "string" internally https://github.com/LDeakin/zarrs/blob/0532fe983b7b42b59dbf84e50a2fe5e6f7bad4ce/zarrs_metadata/src/v2_to_v3.rs#L288
            dtype = String::from("string");
        }
        let fill_value: Bound<'_, PyAny> = chunk_spec.getattr("fill_value")?;
        let fill_value_bytes = fill_value_to_bytes(&dtype, &fill_value)?;
        Ok(Self {
            store,
            key: StoreKey::new(path).map_py_err::<PyValueError>()?,
            representation: get_chunk_representation(chunk_shape, &dtype, fill_value_bytes)?,
        })
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct WithSubset {
    pub item: Basic,
    pub chunk_subset: ArraySubset,
    pub subset: ArraySubset,
}

#[gen_stub_pymethods]
#[pymethods]
impl WithSubset {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        item: Basic,
        chunk_subset: Vec<Bound<'_, PySlice>>,
        subset: Vec<Bound<'_, PySlice>>,
        shape: Vec<u64>,
    ) -> PyResult<Self> {
        let chunk_subset =
            selection_to_array_subset(&chunk_subset, &item.representation.shape_u64())?;
        let subset = selection_to_array_subset(&subset, &shape)?;
        Ok(Self {
            item,
            chunk_subset,
            subset,
        })
    }
}

impl ChunksItem for Basic {
    fn store_config(&self) -> StoreConfig {
        self.store.clone()
    }
    fn key(&self) -> &StoreKey {
        &self.key
    }
    fn representation(&self) -> &ChunkRepresentation {
        &self.representation
    }
}

impl ChunksItem for WithSubset {
    fn store_config(&self) -> StoreConfig {
        self.item.store.clone()
    }
    fn key(&self) -> &StoreKey {
        &self.item.key
    }
    fn representation(&self) -> &ChunkRepresentation {
        &self.item.representation
    }
}

fn get_chunk_representation(
    chunk_shape: Vec<u64>,
    dtype: &str,
    fill_value: Vec<u8>,
) -> PyResult<ChunkRepresentation> {
    // Get the chunk representation
    let data_type = DataType::from_metadata(
        &MetadataV3::new(dtype),
        zarrs::config::global_config().data_type_aliases_v3(),
    )
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
            .map(|(selection, &shape)| slice_to_range(selection, isize::try_from(shape)?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(ArraySubset::new_with_ranges(&chunk_ranges))
    }
}
