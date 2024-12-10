use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, pyclass, Bound, PyAny, PyErr, PyResult};
use pyo3_stub_gen::derive::gen_stub_pyclass;
use zarrs::storage::ReadableWritableListableStorage;

use super::{opendal_builder_to_sync_store, StoreConfig};

#[gen_stub_pyclass]
#[pyclass(extends=StoreConfig)]
pub struct HttpStoreConfig {
    pub root: String,
}

impl HttpStoreConfig {
    pub fn new(path: &str, storage_options: &HashMap<String, Bound<'_, PyAny>>) -> PyResult<Self> {
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

impl TryInto<ReadableWritableListableStorage> for &HttpStoreConfig {
    type Error = PyErr;

    fn try_into(self) -> Result<ReadableWritableListableStorage, Self::Error> {
        let builder = opendal::services::Http::default().endpoint(&self.root);
        opendal_builder_to_sync_store(builder)
    }
}
