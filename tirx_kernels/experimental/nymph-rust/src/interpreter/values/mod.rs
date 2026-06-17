//! The values layer — simulated memory/value types (port of `interpreter/values/`).

pub mod arrays;
pub mod cooperative;
pub mod dtypes;
pub mod indexing;
pub mod ldstmatrix;
pub mod mbars;
pub mod reg_numerics;
pub mod registers;
pub mod runtime;
pub mod scalars;
pub mod smem;
pub mod tcgen05_datapath;
pub mod tensors;
pub mod tmem;

pub use arrays::{coerce_f32_arrayd, ValueArray1, ValueArray2};
pub use dtypes::{is_float_dtype, is_int_dtype, round_scalar, round_to_dtype, wrap_int_to_dtype};
pub use indexing::{flat_index, numel, slice_coords};
pub use mbars::{MbarCell, MbarCellKey, MbarIdentity, MbarValues};
pub use registers::{register_row, RegisterKey, RegisterTensorValue, RegisterValues};
pub use runtime::RuntimeValues;
pub use scalars::ScalarValues;
pub use smem::{dtype_size_bytes, tensor_byte_extent, SmemScratchpad, SmemValues};
pub use tensors::{
    tensor_instance_key, DenseTensorValue, TensorInstanceKey, TensorOwner, TensorValues,
};
pub use tmem::{tmem_layout_for, tmem_physical_range, TmemScratchpad, TmemValues};
