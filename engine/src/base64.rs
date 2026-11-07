//! Standard base64: lenient decoding for skin properties, and the observer key's text form.

const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Skips anything that is not a base64 character, padding included.
pub fn decode(input: &str) -> Vec<u8> {
    let mut table = [-1i32; 256];
    for (i, &c) in CHARS.iter().enumerate() {
        table[c as usize] = i as i32;
    }
    let mut out = Vec::new();
    let (mut val, mut bits) = (0i32, -8i32);
    for c in input.bytes() {
        let t = table[c as usize];
        if t < 0 {
            continue;
        }
        val = (val << 6 | t) & 0xff_ffff;
        bits += 6;
        if bits >= 0 {
            out.push(((val >> bits) & 0xff) as u8);
            bits -= 8;
        }
    }
    out
}

pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let n = group.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for k in 0..4 {
            out.push(if k <= group.len() { CHARS[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        for len in 0..40 {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(decode(&encode(&bytes)), bytes);
        }
        assert_eq!(encode(b"Man"), "TWFu");
        assert_eq!(encode(b"Ma"), "TWE=");
        assert_eq!(encode(b"M"), "TQ==");
    }
}
