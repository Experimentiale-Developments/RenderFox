//! Little-endian binary records and a fast 64-bit hash, for what the engine keeps on disk
//! between runs: the recorded orbit's index and the world it was recorded from.

/// A 64-bit hash of bytes, eight at a time; good enough to tell changed data apart, not
/// meant to resist anyone.
pub fn hash(bytes: &[u8], seed: u64) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = seed ^ (bytes.len() as u64).wrapping_mul(K);
    let (words, rest) = bytes.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(K).rotate_left(29);
    }
    let mut last = [0u8; 8];
    last[..rest.len()].copy_from_slice(rest);
    h = (h ^ u64::from_le_bytes(last)).wrapping_mul(K);
    h ^ (h >> 31)
}

#[derive(Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }
    pub fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i32(&mut self, v: i32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn len(&mut self, n: usize) {
        self.u32(n as u32);
    }
}

pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

pub type Result<T> = std::result::Result<T, String>;

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, at: 0 }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let b = self.bytes.get(self.at..self.at + N).ok_or("truncated record")?;
        self.at += N;
        Ok(b.try_into().unwrap())
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    pub fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take()?))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take()?))
    }
    /// A count, refused if it could not fit in what is left.
    pub fn len(&mut self, each: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(each.max(1)) > self.bytes.len() - self.at {
            return Err("record count past the end".into());
        }
        Ok(n)
    }
    pub fn done(&self) -> bool {
        self.at == self.bytes.len()
    }
}

/// Writes `bytes` to `path` whole or not at all: a crash midway leaves the old file.
pub fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_data()?;
    drop(f);
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let mut w = Writer::default();
        w.u8(7);
        w.i32(-5);
        w.u64(u64::MAX - 3);
        w.f32(1.5);
        w.len(2);
        let mut r = Reader::new(&w.bytes);
        assert_eq!((r.u8().unwrap(), r.i32().unwrap(), r.u64().unwrap()), (7, -5, u64::MAX - 3));
        assert_eq!(r.f32().unwrap(), 1.5);
        assert!(r.len(1000).is_err(), "a count past the end is refused");
        assert!(Reader::new(&[1, 2]).u32().is_err());
    }

    #[test]
    fn hash_tells_bytes_apart() {
        let a = vec![3u8; 4096];
        let mut b = a.clone();
        b[4095] = 4;
        assert_ne!(hash(&a, 1), hash(&b, 1));
        assert_ne!(hash(&a, 1), hash(&a, 2));
        assert_ne!(hash(&a[..7], 1), hash(&a[..8], 1));
        assert_eq!(hash(&a, 9), hash(&a.clone(), 9));
    }
}
