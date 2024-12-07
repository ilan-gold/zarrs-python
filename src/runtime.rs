use std::sync::OnceLock;
use tokio::runtime::Runtime;
use zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

pub fn tokio_block_on() -> TokioBlockOn {
    let runtime = RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"));
    TokioBlockOn(runtime.handle().clone())
}
