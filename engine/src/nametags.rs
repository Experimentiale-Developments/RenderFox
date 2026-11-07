//! Name tags over players, drawn as the game draws them: the name in the game's own
//! pixel font, white on a quarter-opaque black box, dimmed where the player is behind
//! something or sneaking. The game sizes them in the world, which at the stream's
//! distance would be three pixels tall, so here a font pixel is never smaller than two
//! screen pixels, and always a whole number of them, so the letters stay crisp.

use crate::assets::load_png_rgba;

/// One tag to draw: where its bottom edge sits on screen, how many screen pixels a font
/// pixel takes, and how fully its text shows: 1 in the open, 0 hidden or sneaking.
pub struct Tag {
    pub x: f32,
    pub y: f32,
    pub scale: i32,
    pub text: String,
    pub shown: f32,
}

pub struct NameTags {
    rows: [[u8; 8]; 128], // each glyph's rows, a bit per column
    advance: [i32; 128],
}

impl NameTags {
    /// `ascii` is the game's font/ascii.png: 16 x 16 glyphs of 8 x 8.
    pub fn new(ascii: &str) -> Result<NameTags, String> {
        let (w, h, px) = load_png_rgba(ascii).ok_or_else(|| format!("cannot read the name tag font {ascii}"))?;
        if w != 128 || h != 128 {
            return Err(format!("{ascii} is {w} x {h}, not the 128 x 128 font"));
        }
        let mut t = NameTags { rows: [[0; 8]; 128], advance: [0; 128] };
        for c in 0..128usize {
            let (gx, gy) = (c % 16 * 8, c / 16 * 8);
            let mut width = 0;
            for y in 0..8 {
                for x in 0..8 {
                    if px[((gy + y) * 128 + gx + x) * 4 + 3] != 0 {
                        t.rows[c][y] |= 1 << x;
                        width = width.max(x as i32 + 1);
                    }
                }
            }
            // As the game: a glyph advances its width plus one, a space four.
            t.advance[c] = if c == b' ' as usize { 4 } else { width + 1 };
        }
        Ok(t)
    }

    fn glyphs<'a>(&self, text: &'a str) -> impl Iterator<Item = usize> + 'a {
        text.bytes().filter(|&b| b < 128).map(usize::from)
    }

    /// The pixels [x0, y0, x1, y1) a tag covers.
    pub fn bounds(&self, t: &Tag) -> [i32; 4] {
        let s = t.scale;
        let width: i32 = self.glyphs(&t.text).map(|c| self.advance[c]).sum();
        let (bw, bh) = ((width + 2) * s, 10 * s);
        let left = (t.x - bw as f32 / 2.0).round() as i32;
        let top = t.y.round() as i32 - bh;
        [left, top, left + bw, top + bh]
    }

    /// Draws the tags, in the order given, onto 8-bit sRGB planes of width w.
    pub fn draw(&self, planes: &mut [Vec<u8>; 3], w: i32, h: i32, tags: &[Tag]) {
        for t in tags {
            let s = t.scale;
            let width: i32 = self.glyphs(&t.text).map(|c| self.advance[c]).sum();
            // The box runs a font pixel beyond the text on each side and from one above
            // it to the line's bottom (the game's layout), its bottom edge at (x, y).
            let (bw, bh) = ((width + 2) * s, 10 * s);
            let left = (t.x - bw as f32 / 2.0).round() as i32;
            let top = t.y.round() as i32 - bh;
            let mut fill = |x0: i32, y0: i32, x1: i32, y1: i32, colour: u8, alpha: u32| {
                for y in y0.max(0)..y1.min(h) {
                    for x in x0.max(0)..x1.min(w) {
                        let i = (y * w + x) as usize;
                        for p in planes.iter_mut() {
                            p[i] = ((p[i] as u32 * (255 - alpha) + colour as u32 * alpha + 127) / 255) as u8;
                        }
                    }
                }
            };
            fill(left, top, left + bw, top + bh, 0, 64);
            let alpha = 32 + (223.0 * t.shown.clamp(0.0, 1.0)).round() as u32;
            let mut pen = left + s;
            for c in self.glyphs(&t.text) {
                for (row, bits) in self.rows[c].iter().enumerate() {
                    for col in 0..8 {
                        if bits & 1 << col != 0 {
                            let (x, y) = (pen + col * s, top + s + row as i32 * s);
                            fill(x, y, x + s, y + s, 255, alpha);
                        }
                    }
                }
                pen += self.advance[c] * s;
            }
        }
    }
}
