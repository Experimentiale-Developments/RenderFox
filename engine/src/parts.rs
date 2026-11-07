//! Disjoint mutable pieces of one buffer, for the jobs of a parallel pass. Piece `i`
//! belongs to job `i` alone, so its lock is never contended: it exists only so that a
//! closure shared by every worker can hand out `&mut` access without unsafe code.

use std::sync::{Mutex, MutexGuard};

pub struct Parts<'a, T>(Vec<Mutex<&'a mut [T]>>);

impl<'a, T: Send> Parts<'a, T> {
    /// Pieces of `len` elements, the last one shorter if need be.
    pub fn chunks(buf: &'a mut [T], len: usize) -> Self {
        Parts(buf.chunks_mut(len).map(Mutex::new).collect())
    }

    /// Consecutive pieces, each ending at the next of `ends` (increasing offsets).
    pub fn split(mut buf: &'a mut [T], ends: impl IntoIterator<Item = usize>) -> Self {
        let mut pieces = Vec::new();
        let mut at = 0;
        for end in ends {
            let (head, tail) = std::mem::take(&mut buf).split_at_mut(end - at);
            pieces.push(Mutex::new(head));
            buf = tail;
            at = end;
        }
        Parts(pieces)
    }

    /// Rows of `row` elements in `bands` bands, band b holding rows
    /// b * rows / bands up to (b + 1) * rows / bands.
    pub fn bands(buf: &'a mut [T], row: usize, rows: usize, bands: usize) -> Self {
        Self::split(buf, (1..=bands).map(|b| b * rows / bands * row))
    }

    pub fn take(&self, i: usize) -> MutexGuard<'_, &'a mut [T]> {
        self.0[i].try_lock().expect("each piece belongs to exactly one job")
    }
}
