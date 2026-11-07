// Links the patched x264 that x264/build.sh builds as a static library, and
// compiles the glue that opens it (src/x264glue.c). No build dependency: cc and ar are
// already needed to build x264 itself.
use std::path::PathBuf;
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed");
}

fn main() {
    let x264 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../x264/build");
    let lib = x264.join("lib/libx264.a");
    assert!(lib.exists(), "x264 is not built yet: run x264/build.sh first");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("x264glue.o");
    run(Command::new("cc").args(["-O2", "-fPIC", "-Wall", "-Wextra", "-c", "src/x264glue.c", "-o"]).arg(&obj).arg("-I").arg(x264.join("include")));
    run(Command::new("ar").arg("crs").arg(out.join("libx264glue.a")).arg(&obj));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=x264glue");
    println!("cargo:rustc-link-search=native={}", x264.join("lib").display());
    println!("cargo:rustc-link-lib=static=x264");
    println!("cargo:rustc-link-lib=dylib=m");
    // libavcodec's H.264 decoder (src/avglue.c), from the system's ffmpeg: overlay mode
    // decodes the recorded orbit to draw players over it.
    let pkg = |args: &[&str]| -> Vec<String> {
        let out = Command::new("pkg-config").args(args).arg("libavcodec").arg("libavutil").output()
            .unwrap_or_else(|e| panic!("cannot run pkg-config: {e}"));
        assert!(out.status.success(), "pkg-config cannot find libavcodec: install libavcodec-dev");
        String::from_utf8_lossy(&out.stdout).split_whitespace().map(String::from).collect()
    };
    let av = out.join("avglue.o");
    run(Command::new("cc").args(["-O2", "-fPIC", "-Wall", "-Wextra", "-c", "src/avglue.c", "-o"]).arg(&av).args(pkg(&["--cflags"])));
    run(Command::new("ar").arg("crs").arg(out.join("libavglue.a")).arg(&av));
    println!("cargo:rustc-link-lib=static=avglue");
    for flag in pkg(&["--libs"]) {
        if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
        } else if let Some(dir) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rerun-if-changed=src/avglue.c");
    println!("cargo:rerun-if-changed=src/x264glue.c");
    println!("cargo:rerun-if-changed={}", lib.display());
}
