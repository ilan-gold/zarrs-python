use std::sync::Arc;

use pyo3::{exceptions::PyValueError, PyResult};
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

impl CodecPipelineStoreHTTP {
    pub fn new(url_root: &str) -> PyResult<Self> {
        let builder = opendal::services::Http::default().endpoint(url_root);
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

    fn chunk_path(&self, store_path: &str) -> PyResult<String> {
        Ok(store_path.to_string())
    }
}
