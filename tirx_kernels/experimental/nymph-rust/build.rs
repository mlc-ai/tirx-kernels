use std::env;

// Link a multithreaded OpenBLAS for the MMA hot path (cblas_sgemm) — the same
// library the perf prototype linked. Default to the asplos conda env's OpenBLAS;
// override the directory with BLAS_LIB_DIR. We symlink the versioned `.so.0` to
// the unversioned name in OUT_DIR so `-lopenblas` resolves, and embed an rpath to
// the real directory so the loaded extension finds it at runtime.
fn main() {
    let blas_dir = env::var("BLAS_LIB_DIR")
        .unwrap_or_else(|_| "/home/bohanhou/micromamba/envs/asplos/lib".to_string());
    let out_dir = env::var("OUT_DIR").unwrap();
    let link = format!("{out_dir}/libopenblas.so");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(format!("{blas_dir}/libopenblas.so.0"), &link)
        .expect("symlink libopenblas.so");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{blas_dir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{blas_dir}");
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
}
