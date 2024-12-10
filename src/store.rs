use std::{collections::HashMap, sync::Arc};

use opendal::Builder;
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    pyclass,
    types::{PyAnyMethods, PyStringMethods, PyTypeMethods},
    Bound, FromPyObject, PyAny, PyErr, PyResult,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum};

pub use filesystem::FilesystemStoreConfig;
pub use http::HttpStoreConfig;
use zarrs::storage::{
    storage_adapter::async_to_sync::AsyncToSyncStorageAdapter, ReadableWritableListableStorage,
};

use crate::{runtime::tokio_block_on, utils::PyErrExt};

mod filesystem;
mod http;

#[gen_stub_pyclass]
#[pyclass(subclass)]
pub struct StoreConfig;

#[gen_stub_pyclass_enum]
pub enum StoreConfigType {
    Filesystem(FilesystemStoreConfig),
    Http(HttpStoreConfig),
    // TODO: Add support for more stores
}

impl<'py> FromPyObject<'py> for StoreConfigType {
    fn extract_bound(store: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = store.get_type().name()?;
        let name = name.to_str()?;
        match name {
            "LocalStore" => {
                let root: String = store.getattr("root")?.call_method0("__str__")?.extract()?;
                Ok(StoreConfigType::Filesystem(FilesystemStoreConfig::new(
                    root,
                )))
            }
            "RemoteStore" => {
                let fs = store.getattr("fs")?;
                let fs_name = fs.get_type().name()?;
                let fs_name = fs_name.to_str()?;
                let path: String = store.getattr("path")?.extract()?;
                let storage_options: HashMap<String, Bound<'py, PyAny>> =
                    fs.getattr("storage_options")?.extract()?;
                match fs_name {
                    "HTTPFileSystem" => Ok(StoreConfigType::Http(HttpStoreConfig::new(
                        &path,
                        &storage_options,
                    )?)),
                    _ => Err(PyErr::new::<PyNotImplementedError, _>(format!(
                        "zarrs-python does not support {fs_name} (RemoteStore) stores"
                    ))),
                }
            }
            _ => Err(PyErr::new::<PyNotImplementedError, _>(format!(
                "zarrs-python does not support {name} stores"
            ))),
        }
    }
}

impl TryFrom<&StoreConfigType> for ReadableWritableListableStorage {
    type Error = PyErr;

    fn try_from(value: &StoreConfigType) -> Result<Self, Self::Error> {
        match value {
            StoreConfigType::Filesystem(config) => config.try_into(),
            StoreConfigType::Http(config) => config.try_into(),
        }
    }
}

fn opendal_builder_to_sync_store<B: Builder>(
    builder: B,
) -> PyResult<ReadableWritableListableStorage> {
    let operator = opendal::Operator::new(builder)
        .map_py_err::<PyValueError>()?
        .finish();
    let store = Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator));
    let store = Arc::new(AsyncToSyncStorageAdapter::new(store, tokio_block_on()));
    Ok(store)
}
