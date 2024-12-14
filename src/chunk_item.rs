use std::num::NonZeroU64;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods as _, PySlice, PySliceMethods as _},
    Bound, PyAny, PyErr, PyResult,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zarrs::{
    array::{ChunkRepresentation, DataType, FillValue},
    array_subset::ArraySubset,
    metadata::v3::{array::data_type::DataTypeMetadataV3, MetadataV3},
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

#[gen_stub_pymethods]
#[pymethods]
impl Basic {
    #[new]
    fn new(byte_interface: &Bound<'_, PyAny>, chunk_spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        let store: StoreConfig = byte_interface.getattr("store")?.extract()?;
        let path: String = byte_interface.getattr("path")?.extract()?;

        let chunk_shape = chunk_spec.getattr("shape")?.extract()?;
        let dtype: String = chunk_spec
            .getattr("dtype")?
            .call_method0("__str__")?
            .extract()?;
        let fill_value = chunk_spec
            .getattr("fill_value")?
            .call_method0("tobytes")?
            .extract()?;
        Ok(Self {
            store,
            key: StoreKey::new(path).map_py_err::<PyValueError>()?,
            representation: get_chunk_representation(chunk_shape, &dtype, fill_value)?,
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
