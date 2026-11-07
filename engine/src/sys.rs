//! The few POSIX calls std does not wrap.

use std::fs::File;
use std::os::fd::{AsFd, AsRawFd};
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by SIGTERM and SIGINT: every loop winds down and the process exits cleanly.
pub static QUIT: AtomicBool = AtomicBool::new(false);

const PRIO_PROCESS: i32 = 0;
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const F_SETPIPE_SZ: i32 = 1031;
const M_TRIM_THRESHOLD: i32 = -1;
const M_MMAP_THRESHOLD: i32 = -3;
const MADV_DONTNEED: i32 = 4;
const PAGE: usize = 4096;

unsafe extern "C" {
    fn mallopt(param: i32, value: i32) -> i32;
    fn malloc_trim(pad: usize) -> i32;
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}

extern "C" fn on_quit(_: i32) {
    QUIT.store(true, Ordering::SeqCst);
}

pub fn catch_quit_signals() {
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        signal(SIGTERM, on_quit);
        signal(SIGINT, on_quit);
    }
}

/// Nice value for the calling thread (on Linux, PRIO_PROCESS with 0 means this thread).
pub fn lower_priority(nice: i32) {
    // SAFETY: plain syscall wrapper, no memory involved.
    unsafe {
        setpriority(PRIO_PROCESS, 0, nice);
    }
}

/// Memory freed goes back to the system. glibc serves allocations of 128 KiB and more
/// with mappings of their own, which a free unmaps, but raises that threshold to the size
/// of any such block freed: after the renderer gives back its frame buffers once, buffers
/// that size come from the heap, and freed there they stay in the process. A fixed
/// threshold keeps every large buffer (frames, planes, encoders' pictures) in a mapping of
/// its own, and a lower trim threshold returns the heap's freed top sooner.
pub fn return_freed_memory() {
    // SAFETY: plain calls into the C library, made before any other thread starts.
    unsafe {
        mallopt(M_MMAP_THRESHOLD, 256 << 10);
        mallopt(M_TRIM_THRESHOLD, 4 << 20);
    }
}

/// Gives back whatever the allocator holds free, after a large release.
pub fn trim_memory() {
    // SAFETY: plain call into the C library.
    unsafe {
        malloc_trim(0);
    }
}

/// Numbers a page of zero bytes is a valid run of.
pub trait Plain: Copy {}
impl Plain for f32 {}
impl Plain for u32 {}
impl Plain for u16 {}
impl Plain for u8 {}

/// Hands the whole pages inside `v` back to the system: they read as zeros from then on,
/// and take memory again only once written. For buffers of which only a part is used at a
/// time, and always written before it is read.
pub fn release_pages<T: Plain>(v: &mut [T]) {
    let (start, end) = (v.as_mut_ptr() as usize, v.as_mut_ptr() as usize + std::mem::size_of_val(v));
    let (lo, hi) = (start.next_multiple_of(PAGE), end / PAGE * PAGE);
    if hi > lo {
        // SAFETY: whole pages inside a buffer we hold mutably, of a type for which zero
        // bytes are a valid value; private anonymous or heap memory either way.
        unsafe {
            madvise(lo as *mut std::ffi::c_void, hi - lo, MADV_DONTNEED);
        }
    }
}

/// Unbuffered stdout: whole frames go straight to the pipe.
pub fn stdout_file() -> std::io::Result<File> {
    Ok(File::from(std::io::stdout().as_fd().try_clone_to_owned()?))
}

pub fn grow_pipe(f: &File, bytes: i32) {
    // SAFETY: fcntl on a descriptor we own; failure (not a pipe) is harmless.
    unsafe {
        fcntl(f.as_raw_fd(), F_SETPIPE_SZ, bytes);
    }
}
