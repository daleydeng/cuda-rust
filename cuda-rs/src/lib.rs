use std::mem::MaybeUninit;
use cudart_sys as ffi;

mod cuda_macro;
pub use cuda_macro::*;

mod common;
pub use common::*;

wrap_smart_ptr!(Stream, StreamPtr, ffi::CUstream_st, ffi::cudaStreamDestroy);

impl StreamPtr {
    pub fn create() -> Self {
	let mut out = MaybeUninit::uninit();
	unsafe {
	    ffi::cudaStreamCreate(out.as_mut_ptr()).parse().unwrap();
	    Self::from_raw(out.assume_init())
	}
    }

    pub fn create_with_flags(non_block: bool) -> Self {
	let mut out = MaybeUninit::uninit();
	let flags = if non_block { 1 } else { 0 };
	unsafe {
	    ffi::cudaStreamCreateWithFlags(out.as_mut_ptr(), flags).parse().unwrap();
	    Self::from_raw(out.assume_init())
	}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
	let _ = StreamPtr::create();
	// let _ = Stream::create_with_flags(true);
    }
}
