use cudart_sys as ffi;
use more_asserts::*;
use std::ffi::c_void;
use std::mem::MaybeUninit;
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
            ffi::cudaStreamCreateWithFlags(out.as_mut_ptr(), flags)
                .parse()
                .unwrap();
            Self::from_raw(out.assume_init())
        }
    }
}

pub struct HostMemory {
    pub size: usize,
    ptr: *mut c_void,
}

impl Default for HostMemory {
    fn default() -> Self {
        Self {
            size: 0,
            ptr: std::ptr::null_mut(),
        }
    }
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        unsafe { ffi::cudaFreeHost(self.ptr) };
        self.ptr = std::ptr::null_mut();
        self.size = 0;
    }
}

impl HostMemory {
    pub fn malloc(size: usize) -> Self {
        let mut ptr = MaybeUninit::uninit();
        unsafe {
            ffi::cudaMallocHost(ptr.as_mut_ptr(), size).parse().unwrap();
            Self {
                size,
                ptr: ptr.assume_init(),
            }
        }
    }
}

pub struct DeviceMemory {
    pub size: usize,
    ptr: *mut c_void,
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe { ffi::cudaFree(self.ptr) };
            self.ptr = std::ptr::null_mut();
            self.size = 0;
        }
    }
}

impl DeviceMemory {
    pub fn malloc(size: usize) -> Self {
        let mut ptr = MaybeUninit::uninit();
        unsafe {
            ffi::cudaMalloc(ptr.as_mut_ptr(), size).parse().unwrap();
            Self {
                size,
                ptr: ptr.assume_init(),
            }
        }
    }

    pub unsafe fn copy_from_host(&mut self, src: *const c_void, count: usize) {
        assert_le!(count, self.size);
        ffi::cudaMemcpy(
            self.ptr,
            src,
            count,
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
    }

    pub unsafe fn copy_to_host(&self, dst: *mut c_void, count: usize) {
        assert_le!(count, self.size);
        ffi::cudaMemcpy(
            dst,
            self.ptr,
            count,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
    }

    pub fn copy_from_slice<T>(&mut self, data: &[T]) {
        let count = data.len() * std::mem::size_of::<T>();
        unsafe { self.copy_from_host(data.as_ptr() as *const c_void, count) }
    }

    pub fn copy_to_slice<T>(&self, data: &mut [T]) {
        let count = data.len() * std::mem::size_of::<T>();
        unsafe { self.copy_to_host(data.as_mut_ptr() as *mut c_void, count) }
    }

    pub unsafe fn as_ptr(&self) -> *mut c_void {
        self.ptr
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
