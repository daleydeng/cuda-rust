use cudart_sys as ffi;
use std::ffi::CStr;
use thiserror::Error;

pub trait CudaErrorParser {
    fn parse(self) -> Result<(), CudaError>;
}

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("cuda error: {0} code {1}")]
    RuntimeError(String, i32),
}

impl CudaErrorParser for ffi::cudaError_t {
    fn parse(self) -> Result<(), CudaError> {
        use ffi::cudaError::*;
        let code = self;

        if code == cudaSuccess {
            return Ok(());
        }

        let c_str = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(code)) };

        Err(CudaError::RuntimeError(
            c_str.to_str().unwrap().to_owned(),
            code as i32,
        ))
    }
}
