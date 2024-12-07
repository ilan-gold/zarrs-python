use std::{collections::HashMap, sync::Arc};

use pyo3::{exceptions::PyValueError, pyclass, Bound, PyAny, PyResult};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncStorageAdapter;
use zarrs::storage::ReadableWritableListableStorageTraits;
use zarrs_opendal::AsyncOpendalStore;

use crate::{
    runtime::{tokio_block_on, TokioBlockOn},
    utils::PyErrExt,
    CodecPipelineStore,
};

pub struct CodecPipelineStoreHTTP {
    store: Arc<AsyncToSyncStorageAdapter<AsyncOpendalStore, TokioBlockOn>>,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct HTTPStoreConfig {
    pub root: String,
}

impl HTTPStoreConfig {
    pub fn new<'py>(
        path: &str,
        storage_options: &HashMap<String, Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if !storage_options.is_empty() {
            for storage_option in storage_options.keys() {
                match storage_option.as_str() {
                    // TODO: Add support for other storage options
                    "asynchronous" => {}
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported storage option for HTTPFileSystem: {storage_option}"
                        )));
                    }
                }
            }
        }

        Ok(Self {
            root: path.to_string(),
        })
    }
}

impl CodecPipelineStoreHTTP {
    pub fn new(config: &HTTPStoreConfig) -> PyResult<Self> {
        let builder = opendal::services::Http::default().endpoint(&config.root);
        let operator = opendal::Operator::new(builder)
            .map_py_err::<PyValueError>()?
            .finish();
        let store = Arc::new(zarrs_opendal::AsyncOpendalStore::new(operator));
        let store = Arc::new(AsyncToSyncStorageAdapter::new(store, tokio_block_on()));
        Ok(Self { store })
    }
}

impl CodecPipelineStore for CodecPipelineStoreHTTP {
    fn store(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.store.clone()
    }
}
