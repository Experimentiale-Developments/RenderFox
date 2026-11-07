//! The stream's small print: the station, what is playing, and where to join. Three
//! frosted cards, one across the top and two in the bottom corners, drawn straight onto
//! the encoder's RGB frames. Card text
//! is rasterised once and only again when it changes; the per-frame work is the blur
//! under each card and a blend, well under a millisecond at 720p.

use crate::jobs::Jobs;
use crate::parts::Parts;
use crate::vmath::*;
use crate::wide::{F8, I8};
use ab_glyph::{Font as _, FontVec, GlyphId, OutlineCurve, point};
use ab_glyph_rasterizer::Rasterizer;
use std::sync::Arc;
use std::time::SystemTime;

const MARGIN: f32 = 24.0;
const PAD: f32 = 16.0;
const CARD_H: f32 = 58.0;
const BANNER_H: f32 = 42.0;
const RADIUS: f32 = 11.0;
const SHADOW: f32 = 16.0;
const WHITE: Vec3 = Vec3::new(1.0, 1.0, 1.0);
const TITLE_MAX: f32 = 420.0; // longer titles are cut with an ellipsis

#[derive(Clone, Debug)]
pub struct Text {
    pub station: String,
    pub invite: String,
    pub join_label: String,
    pub address: String,
}

impl Default for Text {
    fn default() -> Text {
        Text {
            station: "RENDER FOX".into(),
            invite: "Join the server to see yourself in the live stream".into(),
            join_label: "JOIN US".into(),
            address: "play.example.com".into(),
        }
    }
}

fn be16(d: &[u8], at: usize) -> u16 {
    d.get(at..at + 2).map_or(0, |b| u16::from_be_bytes([b[0], b[1]]))
}

fn be16s(d: &[u8], at: usize) -> i16 {
    be16(d, at) as i16
}

fn be32(d: &[u8], at: usize) -> u32 {
    d.get(at..at + 4).map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Kerning is read from GPOS pair adjustments exactly as stb_truetype reads it, which is
/// what the station's layout was tuned against: lookup type 2, x advance only.
struct Gpos<'a>(&'a [u8]);

impl Gpos<'_> {
    fn coverage_index(&self, table: usize, glyph: i32) -> i32 {
        let d = self.0;
        match be16(d, table) {
            1 => {
                let (mut l, mut r) = (0i32, be16(d, table + 2) as i32 - 1);
                while l <= r {
                    let m = (l + r) >> 1;
                    let straw = be16(d, table + 4 + 2 * m as usize) as i32;
                    if glyph < straw {
                        r = m - 1;
                    } else if glyph > straw {
                        l = m + 1;
                    } else {
                        return m;
                    }
                }
                -1
            }
            2 => {
                let (mut l, mut r) = (0i32, be16(d, table + 2) as i32 - 1);
                while l <= r {
                    let m = (l + r) >> 1;
                    let rec = table + 4 + 6 * m as usize;
                    let (start, end) = (be16(d, rec) as i32, be16(d, rec + 2) as i32);
                    if glyph < start {
                        r = m - 1;
                    } else if glyph > end {
                        l = m + 1;
                    } else {
                        return be16(d, rec + 4) as i32 + glyph - start;
                    }
                }
                -1
            }
            _ => -1,
        }
    }

    fn glyph_class(&self, table: usize, glyph: i32) -> i32 {
        let d = self.0;
        match be16(d, table) {
            1 => {
                let start = be16(d, table + 2) as i32;
                let count = be16(d, table + 4) as i32;
                if glyph >= start && glyph < start + count {
                    return be16(d, table + 6 + 2 * (glyph - start) as usize) as i32;
                }
                0
            }
            2 => {
                let (mut l, mut r) = (0i32, be16(d, table + 2) as i32 - 1);
                while l <= r {
                    let m = (l + r) >> 1;
                    let rec = table + 4 + 6 * m as usize;
                    let (start, end) = (be16(d, rec) as i32, be16(d, rec + 2) as i32);
                    if glyph < start {
                        r = m - 1;
                    } else if glyph > end {
                        l = m + 1;
                    } else {
                        return be16(d, rec + 4) as i32;
                    }
                }
                // "All glyphs not assigned to a class fall into class 0". (OpenType spec)
                0
            }
            _ => -1,
        }
    }

    fn advance(&self, g1: i32, g2: i32) -> i32 {
        let d = self.0;
        if be16(d, 0) != 1 || be16(d, 2) != 0 {
            return 0;
        }
        let list = be16(d, 8) as usize;
        for i in 0..be16(d, list) as usize {
            let lookup = list + be16(d, list + 2 + 2 * i) as usize;
            if be16(d, lookup) != 2 {
                continue;
            }
            for sti in 0..be16(d, lookup + 4) as usize {
                let table = lookup + be16(d, lookup + 6 + 2 * sti) as usize;
                let coverage = self.coverage_index(table + be16(d, table + 2) as usize, g1);
                if coverage == -1 {
                    continue;
                }
                let (vf1, vf2) = (be16(d, table + 4), be16(d, table + 6));
                if vf1 != 4 || vf2 != 0 {
                    return 0;
                }
                match be16(d, table) {
                    1 => {
                        if coverage >= be16(d, table + 8) as i32 {
                            return 0;
                        }
                        let set = table + be16(d, table + 10 + 2 * coverage as usize) as usize;
                        let (mut l, mut r) = (0i32, be16(d, set) as i32 - 1);
                        while l <= r {
                            let m = (l + r) >> 1;
                            let pair = set + 2 + 4 * m as usize;
                            let straw = be16(d, pair) as i32;
                            if g2 < straw {
                                r = m - 1;
                            } else if g2 > straw {
                                l = m + 1;
                            } else {
                                return be16s(d, pair + 2) as i32;
                            }
                        }
                    }
                    2 => {
                        let c1 = self.glyph_class(table + be16(d, table + 8) as usize, g1);
                        let c2 = self.glyph_class(table + be16(d, table + 10) as usize, g2);
                        let (n1, n2) = (be16(d, table + 12) as i32, be16(d, table + 14) as i32);
                        if c1 < 0 || c1 >= n1 || c2 < 0 || c2 >= n2 {
                            return 0;
                        }
                        return be16s(d, table + 16 + 2 * (c1 * n2) as usize + 2 * c2 as usize) as i32;
                    }
                    _ => return 0,
                }
            }
        }
        0
    }
}

struct Font {
    face: FontVec,
    data: Vec<u8>,
    gpos: usize,
    units_per_em: f32,
}

impl Font {
    fn load(path: &str) -> Result<Font, String> {
        let data = std::fs::read(path).map_err(|e| format!("cannot read font {path}: {e}"))?;
        let face = FontVec::try_from_vec(data.clone()).map_err(|_| format!("not a usable font: {path}"))?;
        let units_per_em = face.units_per_em().ok_or_else(|| format!("no units per em in {path}"))?;
        let gpos = (0..be16(&data, 4) as usize)
            .map(|i| 12 + 16 * i)
            .find(|&rec| data.get(rec..rec + 4) == Some(b"GPOS"))
            .map_or(0, |rec| be32(&data, rec + 8) as usize);
        Ok(Font { face, data, gpos, units_per_em })
    }

    fn scale(&self, px: f32) -> f32 {
        px / self.units_per_em
    }

    fn kern(&self, a: GlyphId, b: GlyphId) -> i32 {
        if self.gpos != 0 {
            Gpos(&self.data[self.gpos..]).advance(a.0 as i32, b.0 as i32)
        } else {
            self.face.kern_unscaled(a, b) as i32
        }
    }
}

fn rounded_rect(px: f32, py: f32, w: f32, h: f32, r: f32) -> f32 {
    let qx = (px - w * 0.5).abs() - (w * 0.5 - r);
    let qy = (py - h * 0.5).abs() - (h * 0.5 - r);
    let ox = fmax(qx, 0.0);
    let oy = fmax(qy, 0.0);
    (ox * ox + oy * oy).sqrt() + fmin(fmax(qx, qy), 0.0) - r
}

/// One box blur pass along rows (step 1) or columns (step = stride), edges clamped.
/// Box blur of an interleaved rgb image along its rows, the edges repeated.
fn blur_across(src: &[f32], dst: &mut [f32], w: usize, radius: i32) {
    let norm = 1.0 / (2 * radius + 1) as f32;
    for (s, d) in src.chunks_exact(w * 3).zip(dst.chunks_exact_mut(w * 3)) {
        let at = |x: i32, ch: usize| s[x.clamp(0, w as i32 - 1) as usize * 3 + ch];
        for ch in 0..3 {
            let mut sum = (-radius..=radius).map(|i| at(i, ch)).sum::<f32>();
            for x in 0..w as i32 {
                d[x as usize * 3 + ch] = sum * norm;
                sum += at(x + radius + 1, ch) - at(x - radius, ch);
            }
        }
    }
}

/// The same down the columns, taken a row at a time so every read is sequential.
fn blur_down(src: &[f32], dst: &mut [f32], w: usize, h: usize, radius: i32) {
    let norm = 1.0 / (2 * radius + 1) as f32;
    let row = |y: i32| &src[y.clamp(0, h as i32 - 1) as usize * w * 3..][..w * 3];
    let mut sum = vec![0.0f32; w * 3];
    for i in -radius..=radius {
        for (a, b) in sum.iter_mut().zip(row(i)) {
            *a += b;
        }
    }
    for (y, d) in dst.chunks_exact_mut(w * 3).enumerate() {
        let (add, sub) = (row(y as i32 + radius + 1), row(y as i32 - radius));
        for (((o, a), p), q) in d.iter_mut().zip(sum.iter_mut()).zip(add).zip(sub) {
            *o = *a * norm;
            *a += p - q;
        }
    }
}

/// What is under a card, blurred at half resolution, for its frosted glass.
#[derive(Default)]
struct Frost {
    a: Vec<f32>, // rgb interleaved, the result
    b: Vec<f32>,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
}

#[derive(Default)]
struct Card {
    x: i32, // panel rectangle in pixels
    y: i32,
    w: i32,
    h: i32,
    // Shape, fixed per card, over the panel plus the shadow margin on every side: panel
    // coverage, drop shadow strength and the light rim, 0..1, and the text as
    // premultiplied red, green, blue and alpha planes.
    margin: i32,
    ink: [Vec<f32>; 4],
    inside: Vec<f32>,
    shadow: Vec<f32>,
    rim: Vec<f32>,
}

pub struct Overlay {
    jobs: Arc<Jobs>,
    w: i32,
    h: i32,
    s: f32, // layout scale, 1 at 720p
    text: Text,
    semibold: Font,
    medium: Font,
    now_playing_file: String,
    now_playing_stamp: Option<SystemTime>,
    next_poll: f64,
    station: Card,
    join: Card,
    playing: Card,
    playing_next: Card,
    playing_title: String,
    has_playing: bool,
    has_next: bool,
    fade_start: f64,
    frost: [Frost; 3],
}

impl Overlay {
    pub fn new(jobs: Arc<Jobs>, font_dir: &str, now_playing_file: &str, width: i32, height: i32, text: Text)
               -> Result<Overlay, String> {
        let mut o = Overlay {
            jobs,
            w: width,
            h: height,
            s: height as f32 / 720.0,
            text,
            semibold: Font::load(&format!("{font_dir}/Inter-SemiBold.ttf"))?,
            medium: Font::load(&format!("{font_dir}/Inter-Medium.ttf"))?,
            now_playing_file: now_playing_file.to_string(),
            now_playing_stamp: None,
            next_poll: 0.0,
            station: Card::default(),
            join: Card::default(),
            playing: Card::default(),
            playing_next: Card::default(),
            playing_title: String::new(),
            has_playing: false,
            has_next: false,
            fade_start: -1.0,
            frost: Default::default(),
        };
        o.layout_static();
        Ok(o)
    }

    fn make_card(&self, w: i32, h: i32) -> Card {
        let s = self.s;
        let r = RADIUS * s;
        let drop = 3.0 * s;
        let margin = (SHADOW * s).ceil() as i32;
        let (rw, rh) = (w + 2 * margin, h + 2 * margin);
        let n = (rw * rh) as usize;
        let mut c = Card {
            w,
            h,
            ink: std::array::from_fn(|_| vec![0.0; n]),
            margin,
            inside: vec![0.0; n],
            shadow: vec![0.0; n],
            rim: vec![0.0; n],
            ..Card::default()
        };
        for y in 0..rh {
            for x in 0..rw {
                let px = (x - margin) as f32 + 0.5;
                let py = (y - margin) as f32 + 0.5;
                let d = rounded_rect(px, py, w as f32, h as f32, r);
                let i = (y * rw + x) as usize;
                c.inside[i] = clamp01(0.5 - d);
                c.shadow[i] = 0.24 * (1.0 - smoothstep(-6.0 * s, SHADOW * s, rounded_rect(px, py - drop, w as f32, h as f32, r)));
                c.rim[i] = clamp01(1.0 - (d + 0.75).abs()) * 0.09;
            }
        }
        c
    }

    fn text_width(&self, f: &Font, size: f32, text: &str, tracking: f32) -> f32 {
        let scale = f.scale(size * self.s);
        let mut pen = 0.0f32;
        let mut prev = GlyphId(0);
        for ch in text.chars() {
            let g = f.face.glyph_id(ch);
            if prev.0 != 0 {
                pen += f.kern(prev, g) as f32 * scale;
            }
            pen += f.face.h_advance_unscaled(g) * scale + tracking * self.s;
            prev = g;
        }
        pen - if prev.0 != 0 { tracking * self.s } else { 0.0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_text(&self, c: &mut Card, f: &Font, size: f32, x: f32, baseline: f32, text: &str, colour: Vec3, alpha: f32,
                 tracking: f32) {
        let scale = f.scale(size * self.s);
        let mut pen = x * self.s;
        let base = (baseline * self.s).round() as i32;
        let mut prev = GlyphId(0);
        for ch in text.chars() {
            let g = f.face.glyph_id(ch);
            if prev.0 != 0 {
                pen += f.kern(prev, g) as f32 * scale;
            }
            let ix = pen.floor() as i32;
            let frac = pen - ix as f32;
            if let Some(outline) = f.face.outline(g) {
                // The glyph's box as stb_truetype places it: pixels as little squares,
                // y down.
                let b = outline.bounds;
                let x0 = (b.min.x * scale + frac).floor() as i32;
                let y0 = (-b.min.y * scale).floor() as i32;
                let x1 = (b.max.x * scale + frac).ceil() as i32;
                let y1 = (-b.max.y * scale).ceil() as i32;
                let (gw, gh) = (x1 - x0, y1 - y0);
                if gw > 0 && gh > 0 {
                    let to_px = |p: ab_glyph::Point| point(p.x * scale + frac - x0 as f32, -p.y * scale - y0 as f32);
                    let mut raster = Rasterizer::new(gw as usize, gh as usize);
                    for curve in &outline.curves {
                        match *curve {
                            OutlineCurve::Line(a, b) => raster.draw_line(to_px(a), to_px(b)),
                            OutlineCurve::Quad(a, b, c) => raster.draw_quad(to_px(a), to_px(b), to_px(c)),
                            OutlineCurve::Cubic(a, b, c, d) => raster.draw_cubic(to_px(a), to_px(b), to_px(c), to_px(d)),
                        }
                    }
                    raster.for_each_pixel(|i, coverage| {
                        let (gx, gy) = ((i % gw as usize) as i32, (i / gw as usize) as i32);
                        let (cx, cy) = (ix + x0 + gx, base + y0 + gy);
                        if cx < 0 || cx >= c.w || cy < 0 || cy >= c.h {
                            return;
                        }
                        let byte = ((coverage * 255.0 + 0.5) as i32).min(255);
                        let a = byte as f32 / 255.0 * alpha;
                        if a <= 0.0 {
                            return;
                        }
                        let k = ((cy + c.margin) * (c.w + 2 * c.margin) + cx + c.margin) as usize;
                        for (p, v) in c.ink.iter_mut().zip([colour.x, colour.y, colour.z, 1.0]) {
                            p[k] = v * a + p[k] * (1.0 - a);
                        }
                    });
                }
            }
            pen += f.face.h_advance_unscaled(g) * scale + tracking * self.s;
            prev = g;
        }
    }

    /// A white bar in layout units, its edges antialiased by how much of each pixel it covers.
    fn draw_rule(&self, c: &mut Card, x: f32, y: f32, w: f32, h: f32, alpha: f32) {
        let s = self.s;
        let (x0, x1, y0, y1) = (x * s, (x + w) * s, y * s, (y + h) * s);
        let rw = c.w + 2 * c.margin;
        for py in y0.floor() as i32..y1.ceil() as i32 {
            let cy = fmin(y1, py as f32 + 1.0) - fmax(y0, py as f32);
            for px in x0.floor() as i32..x1.ceil() as i32 {
                let a = cy * (fmin(x1, px as f32 + 1.0) - fmax(x0, px as f32)) * alpha;
                let k = ((py + c.margin) * rw + px + c.margin) as usize;
                for plane in &mut c.ink {
                    plane[k] = a + plane[k] * (1.0 - a);
                }
            }
        }
    }

    fn fit_text(&self, f: &Font, size: f32, text: &str, max_width: f32) -> String {
        if self.text_width(f, size, text, 0.0) <= max_width * self.s {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        for n in (1..=chars.len()).rev() {
            let mut cut: String = chars[..n].iter().collect();
            cut.truncate(cut.trim_end_matches(' ').len());
            cut.push('…');
            if self.text_width(f, size, &cut, 0.0) <= max_width * self.s {
                return cut;
            }
        }
        "…".into()
    }

    fn layout_static(&mut self) {
        let s = self.s;
        // Station, top centre: one line, the name in white, a hairline, and the invitation in
        // grey.
        let label_w = self.text_width(&self.semibold, 11.0, &self.text.station, 1.3) / s;
        let invite_w = self.text_width(&self.medium, 15.0, &self.text.invite, 0.0) / s;
        let (gap, rule) = (14.0, 1.0);
        let invite_x = 1.25 * PAD + label_w + 2.0 * gap + rule;
        let mut station = self.make_card(((invite_x + invite_w + 1.25 * PAD) * s).ceil() as i32, (BANNER_H * s) as i32);
        station.x = (self.w - station.w) / 2;
        station.y = (MARGIN * s) as i32;
        self.draw_text(&mut station, &self.semibold, 11.0, 1.25 * PAD, 26.5, &self.text.station, WHITE, 0.96, 1.3);
        self.draw_rule(&mut station, 1.25 * PAD + label_w + gap, 0.5 * BANNER_H - 8.0, rule, 16.0, 0.28);
        self.draw_text(&mut station, &self.medium, 15.0, invite_x, 26.5, &self.text.invite, WHITE, 0.66, 0.0);
        self.station = station;

        // Join, bottom right: the address is the one thing meant to be read from across a room.
        let label_w = self.text_width(&self.semibold, 10.5, &self.text.join_label, 1.3) / s;
        let addr_w = self.text_width(&self.semibold, 20.0, &self.text.address, 0.0) / s;
        let mut join = self.make_card(((2.0 * PAD + fmax(label_w, addr_w)) * s).ceil() as i32, (CARD_H * s) as i32);
        join.x = self.w - (MARGIN * s) as i32 - join.w;
        join.y = self.h - (MARGIN * s) as i32 - join.h;
        self.draw_text(&mut join, &self.semibold, 10.5, PAD, 23.0, &self.text.join_label, WHITE, 0.66, 1.3);
        self.draw_text(&mut join, &self.semibold, 20.0, PAD, 45.0, &self.text.address, WHITE, 0.98, 0.0);
        self.join = join;
    }

    fn poll_now_playing(&mut self, time: f64) {
        if self.now_playing_file.is_empty() || time < self.next_poll {
            return;
        }
        self.next_poll = time + 1.0;
        let Ok(stamp) = std::fs::metadata(&self.now_playing_file).and_then(|m| m.modified()) else { return };
        if self.now_playing_stamp == Some(stamp) {
            return;
        }
        self.now_playing_stamp = Some(stamp);
        let content = std::fs::read(&self.now_playing_file).unwrap_or_default();
        let content = String::from_utf8_lossy(&content);
        let title = content.split('\n').next().unwrap_or("").trim_end_matches(['\r', ' ']).to_string();
        if title == self.playing_title || self.fade_start >= 0.0 {
            // A change mid-fade is picked up on the next poll, once this one settles.
            if self.fade_start >= 0.0 {
                self.now_playing_stamp = None;
            }
            return;
        }
        self.has_next = !title.is_empty();
        if self.has_next {
            let s = self.s;
            let label = "NOW PLAYING";
            let shown = self.fit_text(&self.medium, 15.5, &title, TITLE_MAX);
            let w = fmax(self.text_width(&self.semibold, 10.5, label, 1.3), self.text_width(&self.medium, 15.5, &shown, 0.0)) / s;
            let mut card = self.make_card(((2.0 * PAD + w) * s).ceil() as i32, (CARD_H * s) as i32);
            card.x = (MARGIN * s) as i32;
            card.y = self.h - (MARGIN * s) as i32 - card.h;
            self.draw_text(&mut card, &self.semibold, 10.5, PAD, 23.0, label, WHITE, 0.66, 1.3);
            self.draw_text(&mut card, &self.medium, 15.5, PAD, 44.0, &shown, WHITE, 0.96, 0.0);
            self.playing_next = card;
        }
        self.playing_title = title;
        self.fade_start = time;
    }

    /// `rgb` is w x h pixels whose top left corner is `origin` in the frame; it must be
    /// even, keeping the half resolution grid the frame's.
    fn frost(w: i32, h: i32, s: f32, rgb: &[Vec<u8>; 3], c: &Card, origin: (i32, i32), f: &mut Frost) {
        // Done at half resolution (it is a blur, nothing is lost) over the panel plus a
        // border so its edges are honest; two box passes each way are close enough to a
        // gaussian.
        let (cx, cy) = (c.x - origin.0, c.y - origin.1);
        let blur = 2.max((4.5 * s) as i32);
        f.x0 = 0.max(cx / 2 - 2 * blur);
        f.y0 = 0.max(cy / 2 - 2 * blur);
        f.w = ((w / 2).min((cx + c.w) / 2 + 2 * blur + 1) - f.x0) as usize;
        f.h = ((h / 2).min((cy + c.h) / 2 + 2 * blur + 1) - f.y0) as usize;
        let (bw, bh) = (f.w, f.h);
        f.a.resize(bw * bh * 3, 0.0);
        f.b.resize(f.a.len(), 0.0);
        for y in 0..bh {
            let r0 = (2 * (f.y0 as usize + y)) * w as usize + 2 * f.x0 as usize;
            let r1 = r0 + w as usize;
            let d = &mut f.a[y * bw * 3..];
            for x in 0..bw {
                for (ch, p) in rgb.iter().enumerate() {
                    d[x * 3 + ch] =
                        0.25 * (p[r0 + 2 * x] as f32 + p[r0 + 2 * x + 1] as f32 + p[r1 + 2 * x] as f32 + p[r1 + 2 * x + 1] as f32);
                }
            }
        }
        for _ in 0..2 {
            blur_across(&f.a, &mut f.b, bw, blur);
            blur_down(&f.b, &mut f.a, bw, bh, blur);
        }
    }

    fn composite(jobs: &Jobs, w: i32, h: i32, rgb: &mut [Vec<u8>; 3], c: &Card, origin: (i32, i32), f: &Frost, alpha: f32) {
        let tint = [10.0, 14.0, 22.0];
        let (cx, cy) = (c.x - origin.0, c.y - origin.1);
        let m = c.margin;
        let rw = c.w + 2 * m;
        let x0 = 0.max(cx - m);
        let y0 = 0.max(cy - m);
        let x1 = w.min(cx + c.w + m);
        let y1 = h.min(cy + c.h + m);
        let wu = w as usize;
        let rows = rgb.each_mut().map(|p| Parts::chunks(&mut p[y0 as usize * wu..y1 as usize * wu], wu));
        let s = F8::splat;
        let (zero, one, lanes) = (s(0.0), s(1.0), I8::lanes());
        jobs.run(y1 - y0, |r, _| {
            let y = y0 + r;
            let mut px = rows.each_ref().map(|p| p.take(r as usize));
            // This row of the glass: the blur between its two nearest rows, already tinted,
            // as planes with room for the eight-wide reads past the last texel.
            let fy = fclamp((y as f32 - 0.5) * 0.5 - f.y0 as f32, 0.0, f.h as f32 - 1.001);
            let (iy, fy) = (fy as usize, fy - (fy as usize) as f32);
            let glass: [Vec<f32>; 3] = std::array::from_fn(|ch| {
                let at = |row: usize, t: usize| f.a[(row * f.w + t) * 3 + ch];
                let mut g: Vec<f32> = (0..f.w).map(|t| mix(mix(at(iy, t), at(iy + 1, t), fy), tint[ch], 0.46)).collect();
                g.resize(f.w + 9, g[f.w - 1]);
                g
            });
            let row = ((y - cy + m) * rw - cx + m) as isize;
            // Eight pixels at a time; the last group ends on the rect, its lanes already
            // done masked off.
            let mut x = x0;
            while x < x1 {
                let xs = x.min(x1 - 8);
                let fresh = (lanes + I8::splat(xs)).gt(I8::splat(x - 1));
                x = xs + 8;
                let i = (row + xs as isize) as usize;
                let field = |p: &[f32]| F8::load(p[i..i + 8].try_into().unwrap());
                let inside = field(&c.inside);
                let shadow = field(&c.shadow) * (one - inside) * s(alpha);
                let (covered, shaded) = (inside.gt(zero), shadow.ge(s(0.002)));
                if !(covered | shaded).any() {
                    continue;
                }
                let fx = ((lanes + I8::splat(xs)).to_float() - s(0.5)) * s(0.5) - s(f.x0 as f32);
                let fx = fx.max(zero).min(s(f.w as f32 - 1.001));
                let ix = fx.floor();
                let frac = fx - ix;
                let t0 = ix.to_int().to_array()[0] as usize;
                let lo = ix.to_int() - I8::splat(t0 as i32);
                let (k_in, rim) = (inside * s(alpha), field(&c.rim) * s(alpha));
                let (ink_a, keep) = (field(&c.ink[3]) * s(alpha), (covered | shaded) & fresh);
                let xu = xs as usize;
                for (ch, p) in px.iter_mut().enumerate() {
                    let bytes: &mut [u8; 8] = (&mut p[xu..xu + 8]).try_into().unwrap();
                    let old = I8::load_u8(bytes);
                    let texels = F8::load(glass[ch][t0..t0 + 8].try_into().unwrap());
                    let (a, b) = (texels.permute(lo), texels.permute(lo + I8::splat(1)));
                    let glass = (b - a).mul_add(frac, a);
                    let mut v = old.to_float() * (one - shadow);
                    let mut lit = (glass - v).mul_add(k_in, v);
                    lit = (s(255.0) - lit).mul_add(rim, lit);
                    lit = lit.mul_add(one - ink_a, field(&c.ink[ch]) * s(255.0 * alpha));
                    v = F8::select(covered, lit, v);
                    *bytes = I8::select(keep, (v + s(0.5)).min(s(255.0)).to_int(), old).to_bytes();
                }
            }
        });
    }

    /// Where any now playing card can reach, shadow and the blur under it included:
    /// pixels [x0, y0, x1, y1). Nothing else is drawn there.
    pub fn playing_reach(&self) -> [i32; 4] {
        let s = self.s;
        let (w, h) = (((2.0 * PAD + TITLE_MAX) * s).ceil() as i32, (CARD_H * s) as i32);
        let (x, y) = ((MARGIN * s) as i32, self.h - (MARGIN * s) as i32 - h);
        let reach = ((SHADOW * s).ceil() as i32).max(4 * 2.max((4.5 * s) as i32) + 2);
        [(x - reach).max(0), (y - reach).max(0), (x + w + reach).min(self.w), self.h]
    }

    /// The now playing card at `time`: which one shows (the incoming one, or not) and
    /// how strongly, and whether a change of title has finished fading.
    fn fade(&mut self, time: f64) -> (Option<(bool, f32)>, bool) {
        self.poll_now_playing(time);
        // Title changes: the old card fades out, then the new one fades in.
        const HALF: f64 = 0.7;
        let mut playing = self.has_playing.then_some((false, 1.0));
        let mut settled = false;
        if self.fade_start >= 0.0 {
            let t = (time - self.fade_start).max(0.0); // a restarted clock should not strand the fade
            playing = if self.has_playing && t < HALF {
                Some((false, smoothstep(0.0, 1.0, (1.0 - t / HALF) as f32)))
            } else if self.has_next {
                Some((true, smoothstep(0.0, 1.0, ((t - if self.has_playing { HALF } else { 0.0 }) / HALF) as f32)))
            } else {
                None
            };
            settled = t >= if self.has_playing { 2.0 * HALF } else { HALF };
        }
        (playing.filter(|&(_, a)| a > 0.004), settled)
    }

    fn settle(&mut self) {
        self.playing = std::mem::take(&mut self.playing_next);
        self.has_playing = self.has_next;
        self.has_next = false;
        self.fade_start = -1.0;
    }

    /// time is wall-clock seconds; it drives the fades.
    /// Over a frame in red, green and blue planes.
    pub fn draw(&mut self, rgb: &mut [Vec<u8>; 3], time: f64) {
        let (playing, settled) = self.fade(time);
        let playing = playing.map(|(next, a)| (if next { &self.playing_next } else { &self.playing }, a));
        let cards: Vec<(&Card, f32)> = [Some((&self.station, 1.0)), Some((&self.join, 1.0)), playing].into_iter().flatten().collect();
        // The cards never overlap, so what lies under each is blurred side by side first.
        let Overlay { ref jobs, w, h, s, ref mut frost, .. } = *self;
        let frosts = Parts::chunks(&mut frost[..], 1);
        let under = &*rgb;
        jobs.run(cards.len() as i32, |i, _| Self::frost(w, h, s, under, cards[i as usize].0, (0, 0), &mut frosts.take(i as usize)[0]));
        drop(frosts);
        for (&(card, alpha), f) in cards.iter().zip(frost.iter()) {
            Self::composite(jobs, w, h, rgb, card, (0, 0), f, alpha);
        }
        if settled {
            self.settle();
        }
    }

    /// Only the now playing card, onto the w x h pixels at `origin` in the frame: all a
    /// replayed frame needs redrawn. Advances the fade as `draw` does. Returns the
    /// rectangle drawn in, [x0, y0, x1, y1) of those pixels, if the card shows.
    pub fn draw_playing(&mut self, rgb: &mut [Vec<u8>; 3], origin: (i32, i32), w: i32, h: i32, time: f64) -> Option<[i32; 4]> {
        let (playing, settled) = self.fade(time);
        let Overlay { ref jobs, s, playing: ref current, ref playing_next, ref mut frost, .. } = *self;
        let drawn = playing.map(|(next, alpha)| {
            let card = if next { playing_next } else { current };
            Self::frost(w, h, s, rgb, card, origin, &mut frost[0]);
            Self::composite(jobs, w, h, rgb, card, origin, &frost[0], alpha);
            let (x, y, m) = (card.x - origin.0, card.y - origin.1, card.margin);
            [(x - m).max(0), (y - m).max(0), (x + card.w + m).min(w), (y + card.h + m).min(h)]
        });
        if settled {
            self.settle();
        }
        drawn
    }
}

// std::clamp semantics, see render.rs.
fn fclamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}
