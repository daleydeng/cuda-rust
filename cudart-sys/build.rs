use anyhow::Result;
use std::path::PathBuf;
use std::{env, fs};

fn main() -> Result<()>{
    let prefix = env::var("PREFIX").or_else(|_| env::var("CONDA_PREFIX"));
    let s_prefix = prefix?;
    env::set_var("LIBCLANG_PATH", format!("{s_prefix}/lib"));
    let prefix = PathBuf::from(s_prefix);

    let inc_dir = prefix.join("include");

    let bindings = bindgen::builder()
        .header("wrapper.h")
	.clang_arg(format!("-I{}", inc_dir.to_str().expect("to_str failed")))
	.default_enum_style(bindgen::EnumVariation::Rust{
            non_exhaustive: false
	})
        .size_t_is_usize(true)
        .allowlist_function("cu.*")
        .allowlist_type("[Cc][Uu].*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("ENV::OUT_DIR not exists!"));
    let bindings_f = out_path.join("bindings.rs");
    bindings
	.write_to_file(&bindings_f)
        .expect("Couldn't write bindings!");

    fs::create_dir_all("gen")?;
    fs::copy(bindings_f, "gen/bindings.rs")?;

    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}
