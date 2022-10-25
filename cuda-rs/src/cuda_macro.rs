#[macro_export]
macro_rules! wrap_smart_ptr {
    ($name:ident, $ptr_name:ident, $raw_ty:ty, $drop_fn:path) => {
        #[repr(C)]
        pub struct $name {
            raw: $raw_ty,
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    $drop_fn(&mut self.raw);
                }
            }
        }

        #[repr(C)]
        pub struct $ptr_name {
            ptr: *mut $name,
        }

        impl $ptr_name {
            unsafe fn from_raw(ptr: *mut $raw_ty) -> Self {
                Self {
                    ptr: ptr as *mut $name,
                }
            }
        }

        impl std::ops::Deref for $ptr_name {
            type Target = $name;

            fn deref(&self) -> &Self::Target {
                unsafe { self.ptr.as_ref().unwrap() }
            }
        }

        impl std::ops::DerefMut for $ptr_name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { self.ptr.as_mut().unwrap() }
            }
        }
    };
}
