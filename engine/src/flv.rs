//! H.264 in FLV, video only: what the engine hands ffmpeg, which adds the radio's audio
//! and passes the video through untouched.

use std::io::{self, Write};

pub struct Flv<W: Write> {
    out: W,
    scratch: Vec<u8>,
}

impl<W: Write> Flv<W> {
    pub fn new(mut out: W) -> io::Result<Flv<W>> {
        // Signature, version 1, video only, header length; then the empty "previous tag".
        out.write_all(&[b'F', b'L', b'V', 1, 1, 0, 0, 0, 9, 0, 0, 0, 0])?;
        Ok(Flv { out, scratch: Vec::new() })
    }

    /// The decoder configuration (ISO 14496-15), from the encoder's one SPS and PPS.
    pub fn sequence_header(&mut self, sps: &[u8], pps: &[u8]) -> io::Result<()> {
        let mut rec = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
        rec.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        rec.extend_from_slice(sps);
        rec.push(1);
        rec.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        rec.extend_from_slice(pps);
        if matches!(sps[1], 100 | 110 | 122 | 144) {
            // 4:2:0, 8-bit luma and chroma, no SPS extensions.
            rec.extend_from_slice(&[0xfd, 0xf8, 0xf8, 0]);
        }
        self.tag(0, 0, true, 0, &rec)
    }

    /// One picture, in decoding order: `nals` as the encoder gave them, each behind a
    /// 4-byte length.
    pub fn picture(&mut self, dts_ms: i64, cts_ms: i32, keyframe: bool, nals: &[u8]) -> io::Result<()> {
        self.tag(dts_ms, cts_ms, keyframe, 1, nals)
    }

    fn tag(&mut self, dts_ms: i64, cts_ms: i32, keyframe: bool, packet: u8, body: &[u8]) -> io::Result<()> {
        let size = 5 + body.len();
        let ts = dts_ms as u32;
        let b = &mut self.scratch;
        b.clear();
        b.push(9);
        b.extend_from_slice(&(size as u32).to_be_bytes()[1..]);
        b.extend_from_slice(&ts.to_be_bytes()[1..]);
        b.push((ts >> 24) as u8);
        b.extend_from_slice(&[0, 0, 0]);
        b.push(if keyframe { 0x17 } else { 0x27 });
        b.push(packet);
        b.extend_from_slice(&cts_ms.to_be_bytes()[1..]);
        self.out.write_all(b)?;
        self.out.write_all(body)?;
        self.out.write_all(&(11 + size as u32).to_be_bytes())?;
        self.out.flush()
    }
}
