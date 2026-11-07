//! Fixed worker pool with a single parallel-for entry point. The caller thread works
//! too, so a pool of N threads uses N + 1 cores while a job runs.
//!
//! The render box shares its cores with the encoder, so a worker is often descheduled
//! when a job starts. The caller therefore waits only for workers that actually took
//! part in the job, never for one that wakes up after the work is gone: items are
//! claimed from a counter tagged with the job's generation, so a late worker cannot
//! take an item of the next job with the previous job's function.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type JobFn<'a> = dyn Fn(i32, u32) + Sync + 'a;

thread_local! {
    // Set by `here`: jobs this thread starts run on it alone.
    static HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with every job it starts done on this thread, as worker 0: for work too small
/// to be worth waking the pool for (overlay mode's few small rectangles, many passes over
/// them), where the wake-ups cost more than the parallelism saves.
pub fn here<R>(f: impl FnOnce() -> R) -> R {
    let was = HERE.with(|h| h.replace(true));
    let r = f();
    HERE.with(|h| h.set(was));
    r
}

struct State {
    // Erased lifetime: `run` does not return while any worker still holds it.
    job: Option<&'static JobFn<'static>>,
    count: i32,
    active: u32,
    generation: u32,
    stop: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
    next: AtomicU64, // generation << 32 | next index
}

pub struct Jobs {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    // One job at a time: the renderer, the overlay and the colour conversion share the
    // pool from different threads.
    serial: Mutex<()>,
}

impl Jobs {
    pub fn new(workers: u32) -> Jobs {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, count: 0, active: 0, generation: 0, stop: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
            next: AtomicU64::new(0),
        });
        let threads = (0..workers)
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("jobs-{}", i + 1))
                    .spawn(move || worker_loop(&s, i + 1))
                    .expect("spawn job worker")
            })
            .collect();
        Jobs { shared, threads, serial: Mutex::new(()) }
    }

    pub fn size(&self) -> u32 {
        self.threads.len() as u32 + 1
    }

    /// f(index, worker) for index in [0, count); worker is 0..size()-1.
    pub fn run<F: Fn(i32, u32) + Sync>(&self, count: i32, f: F) {
        if count <= 0 {
            return;
        }
        if HERE.with(|h| h.get()) {
            for i in 0..count {
                f(i, 0);
            }
            return;
        }
        let _serial = self.serial.lock().unwrap();
        let dynf: &JobFn<'_> = &f;
        // SAFETY: the reference is cleared below, and this function waits until every
        // worker that copied it has finished, before `f` goes out of scope.
        let erased: &'static JobFn<'static> = unsafe { std::mem::transmute::<&JobFn<'_>, &'static JobFn<'static>>(dynf) };
        let generation;
        {
            let mut st = self.shared.state.lock().unwrap();
            st.generation = st.generation.wrapping_add(1);
            generation = st.generation;
            st.job = Some(erased);
            st.count = count;
            self.shared.next.store((generation as u64) << 32, Ordering::SeqCst);
        }
        self.shared.wake.notify_all();
        drain(&self.shared, dynf, count, generation, 0);
        let mut st = self.shared.state.lock().unwrap();
        st.job = None; // workers arriving from now on skip this job
        while st.active != 0 {
            st = self.shared.done.wait(st).unwrap();
        }
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn drain(shared: &Shared, f: &JobFn<'_>, count: i32, generation: u32, worker: u32) {
    let mut cur = shared.next.load(Ordering::SeqCst);
    loop {
        if (cur >> 32) as u32 != generation || (cur as u32) as i32 >= count {
            return;
        }
        match shared.next.compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => {
                f((cur as u32) as i32, worker);
                cur = shared.next.load(Ordering::SeqCst);
            }
            Err(now) => cur = now,
        }
    }
}

fn worker_loop(shared: &Shared, worker: u32) {
    let mut seen = 0u32;
    loop {
        let (f, count, generation) = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if st.stop {
                    return;
                }
                if st.job.is_some() && st.generation != seen {
                    break;
                }
                st = shared.wake.wait(st).unwrap();
            }
            seen = st.generation;
            st.active += 1;
            (st.job.unwrap(), st.count, st.generation)
        };
        drain(shared, f, count, generation, worker);
        let mut st = shared.state.lock().unwrap();
        st.active -= 1;
        if st.active == 0 {
            shared.done.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn every_index_once() {
        let jobs = Jobs::new(3);
        for round in 0..200 {
            let hits: Vec<AtomicUsize> = (0..97).map(|_| AtomicUsize::new(0)).collect();
            jobs.run(97, |i, w| {
                assert!(w < 4);
                hits[i as usize].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "round {round}");
        }
    }
}
