use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, pyclass, PyResult};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use zarrs::{filesystem::FilesystemStore, storage::ReadableWritableListableStorageTraits};

use crate::{utils::PyErrExt, CodecPipelineStore};

pub struct CodecPipelineStoreFilesystem {
    store: Arc<FilesystemStore>,
}

#[gen_stub_pyclass]
#[pyclass]
pub struct FilesystemStoreConfig {
    root: String,
}

impl FilesystemStoreConfig {
    pub fn new(root: String) -> Self {
        Self { root }
    }
}

impl CodecPipelineStoreFilesystem {
    pub fn new(config: &FilesystemStoreConfig) -> PyResult<Self> {
        let store =
            Arc::new(FilesystemStore::new(config.root.clone()).map_py_err::<PyRuntimeError>()?);
        Ok(Self { store })
    }
}

impl CodecPipelineStore for CodecPipelineStoreFilesystem {
    fn store(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.store.clone()
    }
}
