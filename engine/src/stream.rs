//! The stream: every frame rendered, or replayed from the orbit before (replay.rs), then
//! encoded here and handed on as FLV (flv.rs) at the wall clock's pace, for ffmpeg to add
//! the radio to.
//!
//! Three threads. The render thread works a few frames ahead: it renders a frame, or,
//! when that part of the orbit is recorded and nothing it shows has changed since,
//! prepares its replay: the recorded scene under the now playing card with the card
//! drawn over it.
//! The encoder takes one frame per slot of the wall clock, repeating the last rendered
//! one if a render is late so the video never drifts from the audio, and encodes it,
//! records it, or splices a recorded picture with the card's region encoded afresh. The
//! writer sends each picture a fixed delay after its slot, however it was made, so what
//! ffmpeg reads stays even through every switch.

use crate::flv::Flv;
use crate::jobs::Jobs;
use crate::net::{FeedServer, engine_clock};
use crate::overlay::Overlay;
use crate::decode::Orbit;
use crate::replay::{Cache, Gop, PictureStore, Region, RegionStore, from_yuv420, splice};
use crate::sight::{Change, PLAYER_MARGIN, View};
use crate::vmath::{Vec3, length};
use crate::x264::{sps_slice_fields, Encoder, Picture, Settings, QUANT_OFFSET_AQ, TYPE_AUTO, TYPE_IDR};
use crate::{Config, Entities, Renderer, Scene, mix3, quitting, sys, to_yuv420, unix_now, write_png};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Where each frame of the stream falls in the orbit. Frame n is shown at epoch + n / fps,
/// the epoch on a group boundary, so every group of pictures starts at the same points of
/// the orbit each time round.
#[derive(Clone, Copy)]
struct Timeline {
    gop: i64,
    orbit: i64, // frames in one orbit
    base: i64,  // the epoch in frames since 1970
}

impl Timeline {
    fn slot(&self, n: i64) -> usize {
        (self.base + n).rem_euclid(self.orbit) as usize
    }

    /// Group of pictures and frame within it.
    fn place(&self, n: i64) -> (usize, usize) {
        let s = self.slot(n);
        (s / self.gop as usize, s % self.gop as usize)
    }

    fn groups(&self) -> usize {
        (self.orbit / self.gop) as usize
    }
}

/// A rendered frame that may be recorded: what it shows, and when it was drawn.
struct Rec {
    view: Arc<View>,
    eye: Vec3,
    at: f64,
}

enum Job {
    /// A rendered frame, and what to record it as, if it may be.
    Live { n: i64, yuv: Vec<u8>, record: Option<Rec> },
    /// A frame to replay from `gop`, with its region drawn afresh (empty without a region),
    /// or, `full`, the whole frame drawn afresh of which the zones in `zones` are recoded,
    /// with a qp offset for each macroblock from the frame's recorded quantiser in `quant`
    /// (X264_QUANT_OFFSET_AQ where it has none; empty for none at all).
    Replay { n: i64, gop: Arc<Gop>, region: Vec<u8>, zones: i32, full: bool, quant: Vec<f32> },
}

impl Job {
    fn n(&self) -> i64 {
        match self {
            Job::Live { n, .. } | Job::Replay { n, .. } => *n,
        }
    }
}

struct Queue {
    ready: VecDeque<Job>,
    spare: Vec<Vec<u8>>,
    writer_index: i64,
    // Set when the encoder could not start a replay the render thread began (it came too
    // late for the group's first frame): the render thread renders again.
    refused: bool,
}

/// Why a frame was rendered rather than replayed, for the log.
#[derive(Clone, Copy)]
enum Why {
    Unrecorded, // no recording of this part of the orbit, or a change it shows spoilt it
    Players,    // a player, or its name tag, may be in the picture
    Moved,      // the camera's path has changed here
    Midway,     // a replay starts only where a group of pictures does
}

/// The render thread's choice, frame by frame, between rendering and replaying.
///
/// A part of the orbit is replayed whenever it is recorded, no player could be in its
/// frames, and nothing in the world changed that its frames could show. Every change
/// the world worker makes is tested against each recording's views as it comes, and
/// throws away only the recordings that would show it.
struct Planner {
    replay: Option<Replaying>,
    // The renderer has drawn every frame since `warm_from`, live or, towards the end of a
    // replay, in the background, so it takes over with its history in place.
    rendering: bool,
    restart: bool,
    warm_from: i64,
    warm: i64,
    // The last frame that showed a player or a change: the picture is still easing into
    // it for `settle` frames, and is not recorded until then.
    busy_at: i64,
    settle: i64,
    // The last full frame's view.
    view: Option<Arc<View>>,
    // Changes of the last half minute: a group recorded meanwhile is checked against them
    // when it arrives in the cache.
    log: VecDeque<Change>,
    // Overlay mode draws players from the start of a group: one who comes into view part
    // way through a replayed group waits for the next rather than ending the replay, which
    // could only be followed by a whole frame encoded afresh for the rest of the group.
    players_wait: bool,
}

struct Replaying {
    n0: i64,
    gop: Arc<Gop>,
    stop: Option<i64>, // the last frame to replay, once something changed
}

fn same(a: Vec3, b: Vec3) -> bool {
    length(a - b) < 1e-3
}

impl Planner {
    fn new(warm: i64, settle: i64) -> Planner {
        Planner {
            replay: None,
            rendering: false,
            restart: false,
            warm_from: 0,
            warm,
            busy_at: i64::MIN / 2,
            settle,
            view: None,
            log: VecDeque::new(),
            players_wait: false,
        }
    }

    /// Starts the renderer afresh at frame n.
    fn begin(&mut self, n: i64) {
        (self.rendering, self.restart, self.warm_from) = (true, true, n);
    }

    /// Takes the world worker's changes at frame n: every recording that could show one is
    /// thrown away, and one in view of what is being rendered holds recording back.
    fn take(&mut self, changes: Vec<Change>, cache: &Cache, now: f64, n: i64) {
        if self.rendering && let Some(v) = &self.view && changes.iter().any(|c| c.seen_by(v)) {
            self.busy_at = n;
        }
        self.log.extend(changes);
        let newest = self.log.back().map_or(f64::MIN, |c| c.at);
        let (mut spoilt_count, mut culprit) = (0, None);
        cache.retain(|gop| {
            let from = self.log.partition_point(|c| c.at <= gop.checked());
            let spoilt = self.log.range(from..).find(|c| gop.spoilt_by(c));
            gop.set_checked(newest);
            if let Some(c) = spoilt {
                spoilt_count += 1;
                culprit.get_or_insert_with(|| c.hull.as_ref().map(|h| h.iter().fold(Vec3::ZERO, |a, &p| a + p) * (1.0 / h.len() as f32)));
            }
            spoilt.is_none()
        });
        if let Some(at) = culprit {
            match at {
                Some(p) => eprintln!("replay: a change around {:.0} {:.0} {:.0} spoilt {spoilt_count} recorded groups", p.x, p.y, p.z),
                None => eprintln!("replay: the whole world was redone, {spoilt_count} recorded groups dropped"),
            }
        }
        while self.log.front().is_some_and(|c| c.at < now - 30.0) {
            self.log.pop_front();
        }
    }

    /// After a frame is drawn: its view if it was a full one, and whether a player is in it.
    fn drawn(&mut self, renderer: &Renderer, players: &[[Vec3; 8]], n: i64) {
        if renderer.stats.full {
            self.view = Some(Arc::new(renderer.sight()));
        }
        if let Some(v) = &self.view
            && players.iter().any(|p| v.sees(p, PLAYER_MARGIN, false))
        {
            self.busy_at = n;
        }
    }

    /// The group to replay frame n from, or why to render it.
    fn plan(&mut self, n: i64, eye: Vec3, players: &[[Vec3; 8]], can_start: bool, t: &Timeline, cache: &Cache)
            -> Result<Arc<Gop>, Why> {
        let (g, j) = t.place(n);
        if let Some(r) = &mut self.replay {
            if n < r.n0 + t.gop {
                let mut start = false;
                if r.stop.is_none() {
                    let spoilt = !cache.get(g).is_some_and(|c| Arc::ptr_eq(&c, &r.gop));
                    if spoilt || !same(r.gop.eyes[j], eye) || (r.gop.shows(players) && !self.players_wait) {
                        // Render from the first frame replaying may stop before; the frame
                        // before this one may be it. The renderer starts now, so it has
                        // drawn a frame or two by then.
                        r.stop = (j.saturating_sub(1)..r.gop.cuts.len()).find(|&c| r.gop.cuts[c]).map(|c| r.n0 + c as i64);
                        start = true;
                    } else if j as i64 == t.gop - self.warm {
                        // Near the end of the group, if the next one will be rendered, the
                        // renderer starts in the background.
                        start = cache.get((g + 1) % t.groups()).is_none_or(|next| next.shows(players) && !self.players_wait);
                    }
                }
                let replaying = r.stop.is_none_or(|stop| n <= stop).then(|| r.gop.clone());
                if start && !self.rendering {
                    self.begin(n);
                }
                return replaying.ok_or(Why::Midway);
            }
            self.replay = None;
        }
        let gop = cache.get(g).ok_or(Why::Unrecorded)?;
        if !same(gop.eyes[j], eye) {
            return Err(Why::Moved);
        }
        if gop.shows(players) {
            return Err(Why::Players);
        }
        if j != 0 || !can_start {
            return Err(Why::Midway);
        }
        self.replay = Some(Replaying { n0: n, gop: gop.clone(), stop: None });
        self.rendering = false;
        Ok(gop)
    }

    fn recordable(&self, n: i64) -> bool {
        self.rendering && n - self.warm_from >= self.warm && n - self.busy_at >= self.settle
    }
}

/// The region's pixels in the frame and the buffers its round trips go through.
struct RegionPixels {
    mb: Region,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    store: RegionStore,
    rgb: [Vec<u8>; 3],
    yuv: Vec<u8>,
}

impl RegionPixels {
    fn bytes(&self) -> usize {
        self.w * self.h * 3 / 2
    }

    /// The region of a full frame's planes, as 4:2:0 in `yuv`.
    fn take(&mut self, frame: &[Vec<u8>; 3], frame_w: usize, jobs: &Jobs) {
        for (dst, src) in self.rgb.iter_mut().zip(frame) {
            dst.clear();
            for row in 0..self.h {
                let at = (self.y + row) * frame_w + self.x;
                dst.extend_from_slice(&src[at..at + self.w]);
            }
        }
        self.yuv.resize(self.bytes(), 0);
        to_yuv420(&self.rgb, &mut self.yuv, self.w as i32, self.h as i32, jobs);
    }
}

/// Copies the 4:2:0 samples of pixels [x0, y0, x1, y1) (widened to even) from one w x h
/// frame to another.
fn copy_rect(from: &[u8], to: &mut [u8], w: usize, h: usize, rect: [i32; 4]) {
    let [x0, y0, x1, y1] = rect.map(|v| v as usize);
    let (x0, y0, x1, y1) = (x0 & !1, y0 & !1, (x1 + 1) & !1, (y1 + 1) & !1);
    for y in y0..y1 {
        to[y * w + x0..y * w + x1].copy_from_slice(&from[y * w + x0..y * w + x1]);
    }
    let (cw, ch) = (w / 2, h / 2);
    for plane in [w * h, w * h + cw * ch] {
        for y in y0 / 2..y1 / 2 {
            let at = plane + y * cw;
            to[at + x0 / 2..at + x1 / 2].copy_from_slice(&from[at + x0 / 2..at + x1 / 2]);
        }
    }
}

/// A w x h rectangle at (x, y) (even) of a planar 4:2:0 frame of fw x fh, as its own 4:2:0
/// planes in `part`, and back.
fn get_rect(frame: &[u8], part: &mut Vec<u8>, fw: usize, fh: usize, (x, y, w, h): (usize, usize, usize, usize)) {
    part.resize(w * h * 3 / 2, 0);
    let (fl, fc) = frame.split_at(fw * fh);
    let (fu, fv) = fc.split_at(fw * fh / 4);
    let (pl, pc) = part.split_at_mut(w * h);
    let (pu, pv) = pc.split_at_mut(w * h / 4);
    for row in 0..h {
        pl[row * w..][..w].copy_from_slice(&fl[(y + row) * fw + x..][..w]);
    }
    for row in 0..h / 2 {
        let at = (y / 2 + row) * (fw / 2) + x / 2;
        pu[row * w / 2..][..w / 2].copy_from_slice(&fu[at..][..w / 2]);
        pv[row * w / 2..][..w / 2].copy_from_slice(&fv[at..][..w / 2]);
    }
}

fn put_rect(part: &[u8], frame: &mut [u8], fw: usize, fh: usize, (x, y, w, h): (usize, usize, usize, usize)) {
    let (fl, fc) = frame.split_at_mut(fw * fh);
    let (fu, fv) = fc.split_at_mut(fw * fh / 4);
    let (pl, pc) = part.split_at(w * h);
    let (pu, pv) = pc.split_at(w * h / 4);
    for row in 0..h {
        fl[(y + row) * fw + x..][..w].copy_from_slice(&pl[row * w..][..w]);
    }
    for row in 0..h / 2 {
        let at = (y / 2 + row) * (fw / 2) + x / 2;
        fu[at..][..w / 2].copy_from_slice(&pu[row * w / 2..][..w / 2]);
        fv[at..][..w / 2].copy_from_slice(&pv[row * w / 2..][..w / 2]);
    }
}

/// A test of replay (RF_VERIFY=dir): every replayed frame is also rendered, and compared
/// with what was recorded for its slot, eight pixels to a sample. A recording shown after
/// a change it should have been thrown away for differs by far more than two renders of
/// the same scene do; such frames go to the directory as a pair of PNGs.
struct Verify {
    dir: String,
    thumbs: Vec<Option<Vec<u8>>>,
    worst: u8,
    over: usize,
    frames: usize,
    saved: usize,
}

const THUMB: usize = 8;

impl Verify {
    fn thumb(frame: &[Vec<u8>; 3], w: usize, h: usize) -> Vec<u8> {
        let (tw, th) = (w / THUMB, h / THUMB);
        let mut t = vec![0u8; tw * th];
        for ty in 0..th {
            for tx in 0..tw {
                let mut sum = 0u32;
                for y in ty * THUMB..(ty + 1) * THUMB {
                    for x in tx * THUMB..(tx + 1) * THUMB {
                        let i = y * w + x;
                        sum += (54 * frame[0][i] as u32 + 183 * frame[1][i] as u32 + 19 * frame[2][i] as u32) >> 8;
                    }
                }
                t[ty * tw + tx] = (sum / (THUMB * THUMB) as u32) as u8;
            }
        }
        t
    }

    /// Compares a replayed slot's recording with `frame`, rendered now.
    fn check(&mut self, slot: usize, frame: &[Vec<u8>; 3], w: usize, h: usize) {
        let Some(recorded) = &self.thumbs[slot] else { return };
        let now = Verify::thumb(frame, w, h);
        let diffs: Vec<u8> = recorded.iter().zip(&now).map(|(a, b)| a.abs_diff(*b)).collect();
        let worst = diffs.iter().copied().max().unwrap_or(0);
        let over = diffs.iter().filter(|&&d| d > 24).count();
        self.frames += 1;
        self.worst = self.worst.max(worst);
        self.over += over;
        if over > 8 && self.saved < 40 {
            self.saved += 1;
            let (tw, th) = (w / THUMB, h / THUMB);
            let gray = |t: &[u8]| [t.to_vec(), t.to_vec(), t.to_vec()];
            let _ = write_png(&format!("{}/slot{slot:05}-recorded.png", self.dir), tw as i32, th as i32, &gray(recorded));
            let _ = write_png(&format!("{}/slot{slot:05}-now.png", self.dir), tw as i32, th as i32, &gray(&now));
            eprintln!("verify: slot {slot} differs from its recording in {over} samples (worst {worst})");
        }
    }

    fn report(&mut self) {
        if self.frames > 0 {
            eprintln!("verify: {} replayed frames checked, worst sample difference {}, {} samples over 24", self.frames, self.worst,
                      self.over);
        }
        (self.frames, self.worst, self.over) = (0, 0, 0);
    }
}

/// What went out in the last minute, for the log.
#[derive(Default)]
struct Stats {
    render_ms: AtomicU64,
    rendered: AtomicI64,
    why: [AtomicI64; 4],    // rendered frames by Why
    background: AtomicI64, // frames rendered during a replay, to hand over warm
    replayed: AtomicI64,
    overlaid: AtomicI64,    // recorded frames with players drawn over them
    recoded: AtomicI64,     // of which only the players' bands were encoded again
    overlay_ms: AtomicU64, // and the microseconds that took
    repeats: AtomicI64,
    late: AtomicI64,
    late_max_ms: AtomicU64,
    edits: [AtomicU64; 3], // edits shown: count, summed and longest delay in milliseconds
    encode_us: AtomicU64,  // the encoder thread's time on frames (encoding, recoding, splicing)
    recoded_rows: AtomicI64, // macroblock rows of the zones recoded, summed over frames
    clipped: AtomicI64,      // frames whose drawing went past the zones recoded
    encoded: AtomicI64,
}

impl Stats {
    fn smooth(&self, ms: f64) {
        let v = f64::from_bits(self.render_ms.load(Ordering::Relaxed)) * 0.97 + 0.03 * ms;
        self.render_ms.store(v.to_bits(), Ordering::Relaxed);
    }
}

fn x264_settings(cfg: &Config, region: Option<Region>, gop: i64) -> Settings {
    let rate = cfg.bitrate.to_string();
    let options = vec![
        // docs/youtube-ingest.md: constant bitrate, a closed GOP of 2 s, High profile, two
        // B-frames. superfast with veryfast's mb-tree put back.
        "level=4.1".into(),
        format!("threads={}", cfg.encoder_threads),
        format!("bitrate={rate}"),
        format!("vbv-maxrate={rate}"),
        format!("vbv-bufsize={rate}"),
        format!("keyint={gop}"),
        format!("min-keyint={gop}"),
        "scenecut=0".into(),
        "bframes=2".into(),
        "nal-hrd=cbr".into(),
        "force-cfr=1".into(),
        "mbtree=1".into(),
        "rc-lookahead=10".into(),
        // Fixed rather than content-adaptive, so every encoder writes the same headers.
        "direct=spatial".into(),
        "stitchable=1".into(),
        "colorprim=bt709".into(),
        "transfer=bt709".into(),
        "colormatrix=bt709".into(),
        "fullrange=off".into(),
    ];
    Settings {
        preset: cfg.preset.clone(),
        tune: String::new(),
        profile: "high".into(),
        width: cfg.width,
        height: cfg.height,
        fps: cfg.fps,
        options,
        region: region.map_or([0; 4], |r| [r.x0, r.y0, r.x1, r.y1]),
        region_only: false,
        bands: region.map_or(Vec::new(), |r| bands_for(r, (cfg.height + 15) / 16, cfg.band_rows)),
        window: [0, 0],
    }
}

/// Where the bands are cut, in macroblock rows: around the now playing card's rows, which
/// make a band of their own, and every `rows` above it, the top band (sky, mostly) taking
/// what is left. Each band costs the picture a little (about 0.1 dB at 1080p for a cut
/// through busy ground, nothing in the sky), and lets players be drawn over a recorded
/// group by recoding just the bands they are in.
fn bands_for(r: Region, mb_height: i32, rows: i32) -> Vec<i32> {
    if rows < 3 {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    if r.y1 < mb_height {
        cuts.push(r.y1);
    }
    let mut cut = r.y0;
    while cut >= 3 && cuts.len() < 16 {
        cuts.push(cut);
        cut -= rows;
    }
    cuts.sort_unstable();
    cuts
}

/// The zones (bit z for zone z, as Region::zone_of numbers them) whose rows the pixel rows
/// [y0, y1) touch; the card's zone always.
fn zones_for_rows(bands: &[i32], region: Region, y0: i32, y1: i32) -> i32 {
    let (r0, r1) = (y0.max(0) / 16, (y1 + 15) / 16);
    let mut zones = 1;
    let mut top = 0;
    for (b, &bottom) in bands.iter().chain(std::iter::once(&i32::MAX)).enumerate() {
        if r0 < bottom && r1 > top {
            zones |= 1 << (1 + b);
        }
        top = bottom;
    }
    let _ = region;
    zones
}

enum Out {
    Start { sps: Vec<u8>, pps: Vec<u8>, delay: i64 },
    Picture { d: i64, pts: i64, key: bool, data: Vec<u8> },
}

struct Recording {
    g: usize,
    n0: i64,
    pictures: Vec<Picture>,
    views: Vec<Arc<View>>,
    eyes: Vec<Vec3>,
    from: f64,
}

/// The encoder thread's side: encodes rendered frames, records whole groups of them,
/// and replays recorded ones with the region recoded.
struct Video {
    t: Timeline,
    live_settings: Settings,
    inner_settings: Settings,
    region: Option<Region>,
    region_px: (usize, usize, usize, usize),
    cache: Arc<Cache>,
    live: Option<Encoder>,
    segment_start: bool,
    in_flight: VecDeque<(i64, Option<Rec>)>, // live frames not yet out: n, what to record them as
    recording: Option<Recording>,
    inner: Option<Encoder>,
    inner_window: [i32; 2], // the macroblock rows `inner` was given
    groups: VecDeque<(i64, Arc<Gop>, usize, i32)>, // replayed groups not all out: first frame, group, pictures sent, zones recoded
    bands: Vec<i32>,
    canvas: Vec<u8>,                            // the region's encoder's frame: grey but for the region
    headers: (Vec<u8>, Vec<u8>),
    sent: i64,
    out: SyncSender<Out>,
    pic: Picture,
    spliced: Picture,
    outer: Picture, // a recorded picture, read back
}

impl Video {
    fn new(cfg: &Config, t: Timeline, region: Option<Region>, cache: Arc<Cache>, out: SyncSender<Out>)
           -> Result<Video, String> {
        let live_settings = x264_settings(cfg, region, t.gop);
        let bands = live_settings.bands.clone();
        let mut inner_settings = live_settings.clone();
        // One thread and no mb-tree: it follows the recorded frame types and quantisers,
        // and only the region's few macroblocks need real work.
        // No lookahead or frame type decisions either: its frame types and quantisers are
        // given (2.5 times faster). Deciding types from too little lookahead, x264 would
        // override the given ones.
        inner_settings.options.extend(["threads=1".into(), "mbtree=0".into(), "rc-lookahead=0".into(), "b-adapt=0".into()]);
        // Bands recoded over a decoded recording keep each macroblock's recorded quantiser
        // (Job::Replay's quant), and small deadzones keep the levels the recording already
        // quantised to: at the recording's bits, 0.4 dB below it rather than 2.9 dB
        // (replay.rs recode_quality).
        inner_settings.options.extend(["deadzone-inter=6".into(), "deadzone-intra=3".into()]);
        // And only the region's slices are kept, so nothing outside it is encoded at all.
        inner_settings.region_only = std::env::var_os("RF_REGION_FULL").is_none();
        // Its stream is never sent whole, so no HRD: without, x264 pads every picture of
        // the few slices kept with filler up to the stream's bitrate, written and dropped.
        // A window's SPS may differ from the stream's in what slices do not read.
        if inner_settings.region_only {
            inner_settings.options.retain(|o| o != "nal-hrd=cbr");
            inner_settings.options.push("nal-hrd=none".into());
        }
        let mut live = Encoder::open(&live_settings)?;
        let headers = live.parameter_sets()?;
        let delay = live.max_delayed() as i64 + 4;
        out.send(Out::Start { sps: headers.0.clone(), pps: headers.1.clone(), delay }).map_err(|_| "the writer stopped")?;
        let (w, h) = (cfg.width as usize, cfg.height as usize);
        let mut canvas = vec![128u8; w * h * 3 / 2];
        canvas[..w * h].fill(16);
        Ok(Video {
            t,
            live_settings,
            inner_settings,
            region,
            region_px: region.map_or((0, 0, 0, 0), |r| {
                let (x, y, rw, rh) = r.pixels(cfg.width, cfg.height);
                (x as usize, y as usize, rw as usize, rh as usize)
            }),
            cache,
            live: Some(live),
            segment_start: true,
            in_flight: VecDeque::new(),
            recording: None,
            inner: None,
            inner_window: [0, 0],
            groups: VecDeque::new(),
            canvas,
            headers,
            sent: 0,
            out,
            pic: Picture::default(),
            spliced: Picture::default(),
            outer: Picture::default(),
            bands,
        })
    }

    fn open(&self, settings: &Settings) -> Result<Encoder, String> {
        let mut e = Encoder::open(settings)?;
        let (sps, pps) = e.parameter_sets()?;
        // A window's SPS has its own frame size, which no slice depends on.
        let same = if settings.window == [0, 0] {
            (sps.as_slice(), pps.as_slice()) == (self.headers.0.as_slice(), self.headers.1.as_slice())
        } else {
            pps == self.headers.1 && sps_slice_fields(&sps).is_some() && sps_slice_fields(&sps) == sps_slice_fields(&self.headers.0)
        };
        if !same {
            return Err("an x264 encoder wrote different SPS and PPS from the stream's".into());
        }
        Ok(e)
    }

    /// The macroblock rows the region's encoder is given for recoding `zones`: those of the
    /// zones, top to bottom; less to hold and to copy than the whole frame.
    fn window_for(&self, zones: i32) -> [i32; 2] {
        let Some(r) = self.region.filter(|_| self.inner_settings.region_only) else { return [0, 0] };
        let mb_height = (self.live_settings.height + 15) / 16;
        let mut rows = [i32::MAX, i32::MIN];
        let mut take = |a: i32, b: i32| rows = [rows[0].min(a), rows[1].max(b)];
        if zones & 1 != 0 {
            take(r.y0, r.y1);
        }
        let mut top = 0;
        for (k, &bottom) in self.bands.iter().chain(std::iter::once(&mb_height)).enumerate() {
            if (zones >> (1 + k)) & 1 != 0 {
                take(top, bottom);
            }
            top = bottom;
        }
        if rows[0] >= rows[1] { [0, 0] } else { rows }
    }

    fn planes<'a>(&self, yuv: &'a [u8]) -> ([&'a [u8]; 3], [i32; 3]) {
        let (w, h) = (self.live_settings.width as usize, self.live_settings.height as usize);
        let (luma, chroma) = yuv.split_at(w * h);
        let (u, v) = chroma.split_at(w * h / 4);
        ([luma, u, v], [w as i32, w as i32 / 2, w as i32 / 2])
    }

    fn live(&mut self, n: i64, yuv: &[u8], record: Option<Rec>) -> Result<(), String> {
        self.end_replay()?;
        let (_, j) = self.t.place(n);
        if self.live.is_none() {
            self.live = Some(self.open(&self.live_settings)?);
            self.segment_start = true;
        }
        let kind = if j == 0 || self.segment_start { TYPE_IDR } else { TYPE_AUTO };
        self.segment_start = false;
        self.in_flight.push_back((n, record));
        let frame = self.planes(yuv);
        let slot = self.t.slot(n) as i32;
        let mut pic = std::mem::take(&mut self.pic);
        let out = self.live.as_mut().unwrap().encode(Some(frame), n, kind, -1, slot, 0, &mut pic)?;
        if out {
            self.recorded(&pic);
            self.send(&pic);
        }
        self.pic = pic;
        Ok(())
    }

    /// Keeps a live picture for the recording of its group, which is complete, and goes
    /// in the cache, once all of the group's frames are in it.
    fn recorded(&mut self, pic: &Picture) {
        let record = match self.in_flight.iter().position(|f| f.0 == pic.pts) {
            Some(i) => self.in_flight.remove(i).unwrap().1,
            None => None,
        };
        let (g, j) = self.t.place(pic.pts);
        let frames = self.t.gop as usize;
        if pic.kind == TYPE_IDR && j == 0 {
            self.recording = record.is_some().then(|| Recording {
                g,
                n0: pic.pts,
                pictures: Vec::new(),
                views: Vec::new(),
                eyes: vec![Vec3::ZERO; frames],
                from: f64::MAX,
            });
        }
        let Some(r) = &mut self.recording else { return };
        let Some(rec) = record.filter(|_| r.g == g) else {
            self.recording = None;
            return;
        };
        if !r.views.iter().any(|v| Arc::ptr_eq(v, &rec.view)) {
            r.views.push(rec.view);
        }
        r.eyes[(pic.pts - r.n0) as usize] = rec.eye;
        r.from = r.from.min(rec.at);
        let mut kept = pic.clone();
        kept.pts -= r.n0;
        r.pictures.push(kept);
        if r.pictures.len() == frames {
            let r = self.recording.take().unwrap();
            self.cache.put(r.g, Gop::new(r.pictures, r.views, r.eyes, r.from));
        }
    }

    fn end_live(&mut self) -> Result<(), String> {
        let Some(mut enc) = self.live.take() else { return Ok(()) };
        let mut pic = std::mem::take(&mut self.pic);
        while enc.delayed() > 0 {
            if enc.encode(None, 0, TYPE_AUTO, -1, -1, 0, &mut pic)? {
                self.recorded(&pic);
                self.send(&pic);
            }
        }
        self.pic = pic;
        self.in_flight.clear();
        self.recording = None;
        Ok(())
    }

    fn replay(&mut self, n: i64, gop: Arc<Gop>, region: &[u8], zones: i32, full: bool, quant: &[f32]) -> Result<(), String> {
        self.end_live()?;
        let (_, j) = self.t.place(n);
        if j == 0 {
            self.groups.push_back((n, gop.clone(), 0, zones));
        }
        if self.region.is_none() {
            // Nothing to recode: one recorded picture a frame, in decoding order.
            let (n0, gop, sent, _) = self.groups.back_mut().unwrap();
            let mut pic = std::mem::take(&mut self.outer);
            gop.picture(*sent, &mut pic)?;
            pic.pts += *n0;
            *sent += 1;
            if *sent == gop.len() {
                self.groups.pop_back();
            }
            self.send(&pic);
            self.outer = pic;
            return Ok(());
        }
        let zones = self.groups.back().map_or(1, |g| g.3);
        // A group recoding other zones (it starts with an IDR) gets an encoder of its own
        // rows, once the last one has finished the groups before.
        let window = self.window_for(zones);
        if self.inner.is_some() && window != self.inner_window {
            self.drain_inner();
        }
        if self.inner.is_none() {
            let settings = Settings { window, ..self.inner_settings.clone() };
            self.inner = Some(self.open(&settings)?);
            self.inner_window = window;
        }
        let (kind, qp) = gop.plan[j];
        let slot = self.t.slot(n) as i32;
        if full {
            // The whole frame drawn afresh: its recoded zones are taken from it as it is.
            let frame = self.planes(region);
            let mut pic = std::mem::take(&mut self.pic);
            let quant = Some(quant).filter(|q| !q.is_empty());
            let out = self.inner.as_mut().unwrap().encode_with(Some(frame), n, kind, qp, slot, zones, quant, &mut pic);
            if out? {
                self.splice_out(&pic)?;
            }
            self.pic = pic;
            return Ok(());
        }
        let (x, y, w, h) = self.region_px;
        let fw = self.live_settings.width as usize;
        let fh = self.live_settings.height as usize;
        let (luma, chroma) = self.canvas.split_at_mut(fw * fh);
        let (u, v) = chroma.split_at_mut(fw * fh / 4);
        let (src_y, src_c) = region.split_at(w * h);
        let (src_u, src_v) = src_c.split_at(w * h / 4);
        for row in 0..h {
            luma[(y + row) * fw + x..][..w].copy_from_slice(&src_y[row * w..][..w]);
        }
        for row in 0..h / 2 {
            let at = (y / 2 + row) * (fw / 2) + x / 2;
            u[at..][..w / 2].copy_from_slice(&src_u[row * w / 2..][..w / 2]);
            v[at..][..w / 2].copy_from_slice(&src_v[row * w / 2..][..w / 2]);
        }
        let canvas = std::mem::take(&mut self.canvas);
        let frame = self.planes(&canvas);
        let mut pic = std::mem::take(&mut self.pic);
        let out = self.inner.as_mut().unwrap().encode(Some(frame), n, kind, qp, slot, zones, &mut pic);
        self.canvas = canvas;
        if out? {
            self.splice_out(&pic)?;
        }
        self.pic = pic;
        Ok(())
    }

    /// The recorded picture the region's encoder just matched, with its region swapped in.
    fn splice_out(&mut self, inner: &Picture) -> Result<(), String> {
        let (n0, gop, sent, zones) = self.groups.front_mut().ok_or("the region's encoder gave a picture no group expects")?;
        let zones = *zones;
        let mut outer = std::mem::take(&mut self.outer);
        gop.picture(*sent, &mut outer)?;
        if outer.pts + *n0 != inner.pts || outer.kind != inner.kind {
            return Err(format!("the region's encoder left the recorded frame types at frame {}", inner.pts));
        }
        let mut spliced = std::mem::take(&mut self.spliced);
        let region = self.region.unwrap();
        splice(&outer, inner, |mb| (zones >> region.zone_of(&self.bands, mb)) & 1 != 0, &mut spliced);
        self.outer = outer;
        spliced.pts = inner.pts;
        *sent += 1;
        if *sent == gop.len() {
            self.groups.pop_front();
        }
        self.send(&spliced);
        self.spliced = spliced;
        Ok(())
    }

    /// Drops a replay part way, its pictures not sent: what follows starts afresh with an IDR.
    fn abandon_replay(&mut self) {
        self.inner = None;
        self.groups.clear();
    }

    /// Finishes a replay the render thread stopped after a frame everything so far ends
    /// on, so the region's encoder can finish every picture it holds.
    /// Stopped anywhere else (the overlay given up part way through a group), the B-frames
    /// it holds can only come out as other types than recorded: they are dropped, a frame
    /// or two that the player shows the frame before of, rather than the stream ended.
    fn end_replay(&mut self) -> Result<(), String> {
        self.drain_inner();
        self.groups.clear();
        Ok(())
    }

    /// Finishes every picture the region's encoder holds, and lets it go.
    fn drain_inner(&mut self) {
        if let Some(mut enc) = self.inner.take() {
            let mut pic = std::mem::take(&mut self.pic);
            let mut dropped = 0;
            while enc.delayed() > 0 {
                match enc.encode(None, 0, TYPE_AUTO, -1, -1, 0, &mut pic) {
                    Ok(true) if dropped == 0 => {
                        if let Err(e) = self.splice_out(&pic) {
                            eprintln!("replay: {e}, ending it here");
                            dropped += 1;
                        }
                    }
                    Ok(true) => dropped += 1,
                    Ok(false) => {}
                    Err(e) => {
                        eprintln!("replay: {e}, ending it here");
                        break;
                    }
                }
            }
            if dropped > 0 {
                eprintln!("replay: {dropped} frames dropped");
            }
            self.pic = pic;
        }
    }

    fn send(&mut self, pic: &Picture) {
        let _ = self.out.send(Out::Picture { d: self.sent, pts: pic.pts, key: pic.kind == TYPE_IDR, data: pic.data.clone() });
        self.sent += 1;
    }
}

/// Runs the stream until told to quit.
#[allow(clippy::too_many_arguments)]
pub fn run(cfg: &Config, scene: &mut Scene, renderer: &mut Renderer, entities: &mut Entities, mut overlay: Option<&mut Overlay>,
           jobs: &Jobs, feed: &FeedServer, epoch_file: &str, prerecord: bool) {
    let (ow, oh) = (renderer.width(), renderer.height());
    let frame_bytes = (ow * oh * 3 / 2) as usize;
    let fps = cfg.fps as i64;
    let orbit = (cfg.period as f64 * fps as f64).round() as i64;
    let gop = 2 * fps;
    let gop_frames = gop;
    // The epoch on a group boundary: every group then starts at the same points of the orbit.
    let group_seconds = gop as f64 / fps as f64;
    let timeline = || {
        let epoch = (unix_now() / group_seconds).ceil() * group_seconds;
        (epoch, Timeline { gop, orbit, base: (epoch * fps as f64).round() as i64 })
    };
    let (mut epoch, mut t) = timeline();
    let replayable = cfg.replay && (cfg.period as f64 * fps as f64 - orbit as f64).abs() < 1e-6 && orbit % gop == 0;
    if cfg.replay && !replayable {
        eprintln!("replay: off, the orbit is not a whole number of {group_seconds} s groups");
    }
    // The now playing card changes under the music: its region is recoded on replay.
    let mut region = None;
    if replayable && let Some(o) = overlay.as_deref() {
        let mb = Region::covering(o.playing_reach(), ow, oh);
        let (x, y, w, h) = mb.pixels(ow, oh);
        let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
        match RegionStore::create(&cfg.replay_store, orbit as usize, w * h * 3 / 2, true) {
            Ok(store) => region = Some(RegionPixels { mb, x, y, w, h, store, rgb: Default::default(), yuv: Vec::new() }),
            Err(e) => eprintln!("replay: off, {e}"),
        }
    }
    let replay_on = replayable && (region.is_some() || overlay.is_none());
    if let Some(r) = &region {
        eprintln!("replay: the now playing card's region is macroblocks {},{} to {},{} ({}x{} pixels)", r.mb.x0, r.mb.y0,
                  r.mb.x1, r.mb.y1, r.w, r.h);
    }
    let region_mb = region.as_ref().map(|r| r.mb);
    // A recorded group is about 2.5 MB at 10 Mbit/s; the constant bitrate's buffer lets one
    // run to 3.75 MB, and the slot leaves room past that.
    let pictures = replay_on.then(|| {
        let path = std::path::Path::new(&cfg.replay_store).with_file_name("pictures.bin");
        let slot = ((cfg.bitrate as u64 * 1000 / 8) * 2 * (gop as u64 / fps as u64)).max(8 << 20);
        PictureStore::create(&path.to_string_lossy(), t.groups(), slot, true).map_err(|e| eprintln!("replay: {e}, keeping recordings in memory")).ok()
    });
    let cache = Arc::new(Cache::new(t.groups(), pictures.flatten()));
    // The last run's recordings, if made with these settings by this build: put in the cache
    // once the world worker has compared the world with the one they show.
    let keep_path = std::path::Path::new(&cfg.replay_store).with_file_name("orbit.keep");
    let fingerprint = replay_on.then(|| fingerprint(cfg, region_mb, gop)).flatten();
    let mut kept: Vec<(usize, Gop)> = Vec::new();
    if let Some(fp) = fingerprint {
        match std::fs::read(&keep_path).map_err(|e| e.to_string()).and_then(|b| read_keep(&b, fp, &cache)) {
            Ok((world, groups)) => {
                eprintln!("replay: {} recorded groups kept from the last run, checked against the world once it is in", groups.len());
                *scene.keep.saved.lock().unwrap() = Some(world);
                kept = groups;
            }
            Err(e) if keep_path.exists() => eprintln!("replay: not using the last run's recordings: {e}"),
            Err(_) => {}
        }
    }
    let mut keeper = Keeper { path: keep_path, fingerprint, written: 0, writing: None };
    if prerecord && replay_on {
        while !scene.ready() {
            if quitting() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        adopt_kept(scene, &cache, &mut kept);
        // Recordings hold the world alone: players in view are drawn over them on replay.
        scene.bare = true;
        prerecord_orbit(cfg, scene, renderer, entities, overlay.as_deref_mut(), jobs, t, epoch, region.as_mut(), region_mb, &cache);
        scene.bare = false;
        // Groups start on the same points of the orbit whatever the epoch, so the
        // recordings hold for a stream starting now.
        (epoch, t) = timeline();
    }
    let stats = Arc::new(Stats::default());
    let (tx, rx) = sync_channel::<Out>(4 * gop as usize);

    let dt = 1.0 / cfg.fps as f64;
    let ahead = cfg.ahead as usize;
    let queue = Mutex::new(Queue { ready: VecDeque::new(), spare: Vec::new(), writer_index: -1, refused: false });
    let changed = Condvar::new();

    // Until the world arrives: a sky gradient with the cards on it, not black, so a
    // restart is gentle on screen and the music never plays over nothing.
    let mut waiting: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0u8; (ow * oh) as usize]);
    for y in 0..oh {
        let c = mix3(cfg.look.sky_zenith, cfg.look.sky_horizon, y as f32 / oh as f32);
        for (p, v) in waiting.iter_mut().zip([c.x, c.y, c.z]) {
            p[(y * ow) as usize..((y + 1) * ow) as usize].fill((v.powf(1.0 / 2.2) * 255.0) as u8);
        }
    }

    std::thread::scope(|s| {
        // The render thread.
        s.spawn(|| {
            // Unix time to the engine's clock, which the feed's timestamps are on.
            let engine_offset = engine_clock() - unix_now();
            // Only until the world arrives; then let go.
            let mut waiting = std::mem::take(&mut waiting);
            let mut still = waiting.clone();
            let frames = |seconds: f32| (seconds as f64 * cfg.fps as f64).round() as i64;
            let mut planner = Planner::new(frames(cfg.warm).clamp(1, gop), frames(cfg.settle));
            let mut verify = std::env::var("RF_VERIFY").ok().filter(|_| replay_on).map(|dir| {
                let _ = std::fs::create_dir_all(&dir);
                Verify { dir, thumbs: vec![None; orbit as usize], worst: 0, over: 0, frames: 0, saved: 0 }
            });
            let mut verify_log = Instant::now();
            let mut orbit_frames: Option<Orbit> = None;
            let mut replaying_for = 0i64;
            // The group being drawn over with only some zones encoded again, and those zones.
            let mut recode: Option<(usize, i32)> = None;
            // How far above and below the rows its players started in a group's drawing has
            // strayed, and so how wide the next group's zones are drawn around them.
            let mut excursion = [0i32; 2];
            let mut margin = [48i32; 2];
            let mut start_rows: Option<[i32; 2]> = None;
            let mut overlay_faults = 0;
            let bands = x264_settings(cfg, region_mb, gop_frames).bands;
            let mut overlay_on = replay_on && std::env::var_os("RF_NO_OVERLAY").is_none();
            let mut k: i64 = 0;
            let mut have_world = false;
            while !quitting() {
                if !have_world {
                    have_world = scene.ready();
                    if have_world {
                        waiting = Default::default();
                        still = Default::default();
                    }
                }
                let (mut buf, can_start) = {
                    let mut q = changed.wait_while(queue.lock().unwrap(), |q| q.ready.len() >= ahead && !quitting()).unwrap();
                    if quitting() {
                        break;
                    }
                    // Never make a frame the encoder has already passed.
                    k = k.max(q.writer_index + 1);
                    if std::mem::take(&mut q.refused) {
                        planner.replay = None;
                        // Bands recoded over a group can only start with it.
                        recode = None;
                    }
                    (q.spare.pop().unwrap_or_default(), k > q.writer_index + 1)
                };
                let at = epoch + k as f64 * dt;
                let job = if have_world {
                    let (pose, players) = scene.prepare(renderer, entities, at, at + engine_offset);
                    if replay_on {
                        adopt_kept(scene, &cache, &mut kept);
                        // A snapshot of the world taken before the changes just taken: every
                        // recording that could show one of them is gone from the cache.
                        let snapshot = scene.keep.snapshot.lock().unwrap().take();
                        planner.take(scene.changes.take(), &cache, engine_clock(), k);
                        if let Some(snap) = snapshot {
                            keeper.checkpoint(snap.1, &cache);
                        }
                    }
                    planner.players_wait = overlay_on && verify.is_none();
                    let plan = if replay_on { planner.plan(k, pose.eye, &players, can_start, &t, &cache) } else { Err(Why::Unrecorded) };
                    // Only players differ from a recorded group, or a replay cannot start
                    // here: the recorded frame with the players drawn over it, encoded
                    // afresh, costs a fraction of rendering it.
                    let overlay_gop = match plan {
                        Err(Why::Players | Why::Midway) if overlay_on && verify.is_none() => {
                            let (g, j) = t.place(k);
                            cache.get(g).filter(|gop| same(gop.eyes[j], pose.eye))
                        }
                        _ => None,
                    };
                    if overlay_gop.is_some() {
                        planner.rendering = false;
                    } else if let Err(_) = plan
                        && !planner.rendering
                    {
                        planner.begin(k);
                    }
                    if std::mem::take(&mut planner.restart) {
                        renderer.restart();
                    }
                    // Only replaying, the renderer's buffers long gone and every group
                    // recorded: the world's meshes may go, and its shadow maps with them.
                    let idle = plan.is_ok()
                        && overlay_gop.is_none()
                        && !planner.rendering
                        && verify.is_none()
                        && replaying_for >= 5 * fps
                        && cache.recorded() == t.groups();
                    scene.rest(idle, at + engine_offset);
                    if scene.resting() {
                        renderer.forget_shadows();
                    }
                    if let Some(gop) = overlay_gop {
                        let r0 = Instant::now();
                        // Where the players will be drawn, and so which zones of the recording
                        // this frame needs: with bands, only the bands they are in are decoded
                        // and encoded again, chosen at the start of each group (a little wider
                        // than they are, as they move); a player who leaves them has the rest
                        // of the group decoded and encoded whole.
                        let (bodies, rects, tags) = scene.overlay_plan(renderer, entities, &pose, at, at + engine_offset);
                        let (g, j) = t.place(k);
                        let zones = match region_mb.filter(|_| !bands.is_empty()) {
                            Some(rm) => {
                                let rows = rects.iter().fold(None, |b: Option<[i32; 2]>, r| Some(b.map_or([r[1], r[3]], |b| [b[0].min(r[1]), b[1].max(r[3])])));
                                if j == 0 {
                                    // Players move on much as they have been moving: the zones
                                    // reach as far as the last group's drawing strayed, and more.
                                    if start_rows.is_some() {
                                        margin = excursion.map(|e| (16 + e * 5 / 4).clamp(32, 192));
                                    }
                                    excursion = [0, 0];
                                    start_rows = rows;
                                    let z = rects.iter().fold(1, |z, r| z | zones_for_rows(&bands, rm, r[1] - margin[0], r[3] + margin[1]));
                                    recode = Some((g, z));
                                } else if let (Some(s0), Some(now)) = (start_rows, rows) {
                                    excursion = [excursion[0].max(s0[0] - now[0]), excursion[1].max(now[1] - s0[1])];
                                }
                                match recode {
                                    Some((rg, z)) if rg == g => {
                                        // Drawing past the zones (a player who came into view or
                                        // moved further than foreseen) is left out until the next
                                        // group: never more than two seconds, and far cheaper than
                                        // encoding the rest of the group whole.
                                        if !rects.iter().all(|r| zones_for_rows(&bands, rm, r[1], r[3]) & !z == 0) {
                                            stats.clipped.fetch_add(1, Ordering::Relaxed);
                                        }
                                        z
                                    }
                                    _ => {
                                        recode = None;
                                        -1
                                    }
                                }
                            }
                            None => -1,
                        };
                        let made = (|| -> Result<(), String> {
                            if orbit_frames.is_none() {
                                let settings = x264_settings(cfg, region_mb, gop_frames);
                                let (sps, pps) = Encoder::open(&settings)?.parameter_sets()?;
                                let rm = region_mb.unwrap_or(Region { x0: 0, y0: 0, x1: 0, y1: 0, mb_width: (ow + 15) / 16 });
                                orbit_frames = Some(Orbit::new(&sps, &pps, ow, oh, rm, settings.bands)?);
                            }
                            // The card's own zone comes from the region store below, not the recording.
                            // The card's own zone comes from the region store below, not the
                            // recording; but a picture must keep some slice to be decoded.
                            let wanted = if zones == -1 { -1 } else if zones & !1 != 0 { zones & !1 } else { 1 };
                            orbit_frames.as_mut().unwrap().frame(&gop, j, &mut buf, wanted)
                        })();
                        match made {
                            Ok(()) => {
                                // The recording's card is its own time's: the scene under it
                                // comes back from the store, and the card goes on last.
                                if let Some(r) = &mut region {
                                    r.yuv.resize(r.bytes(), 0);
                                    if r.store.read(t.slot(k), &mut r.yuv).is_ok() {
                                        put_rect(&r.yuv, &mut buf, ow as usize, oh as usize, (r.x, r.y, r.w, r.h));
                                    }
                                }
                                scene.overlay(renderer, &pose, at, at + engine_offset, &mut buf, &bodies, &rects, &tags);
                                for late in scene.latencies.drain(..) {
                                    let ms = (late * 1000.0).max(0.0) as u64;
                                    stats.edits[0].fetch_add(1, Ordering::Relaxed);
                                    stats.edits[1].fetch_add(ms, Ordering::Relaxed);
                                    stats.edits[2].fetch_max(ms, Ordering::Relaxed);
                                }
                                if let (Some(r), Some(o)) = (&mut region, overlay.as_deref_mut()) {
                                    let mut yuv = std::mem::take(&mut r.yuv);
                                    get_rect(&buf, &mut yuv, ow as usize, oh as usize, (r.x, r.y, r.w, r.h));
                                    from_yuv420(&yuv, r.w, r.h, &mut r.rgb);
                                    if let Some(drawn) = o.draw_playing(&mut r.rgb, (r.x as i32, r.y as i32), r.w as i32, r.h as i32, at) {
                                        let mut fresh = vec![0u8; r.bytes()];
                                        to_yuv420(&r.rgb, &mut fresh, r.w as i32, r.h as i32, jobs);
                                        copy_rect(&fresh, &mut yuv, r.w, r.h, drawn);
                                        put_rect(&yuv, &mut buf, ow as usize, oh as usize, (r.x, r.y, r.w, r.h));
                                    }
                                    r.yuv = yuv;
                                }
                                stats.overlaid.fetch_add(1, Ordering::Relaxed);
                                stats.overlay_ms.fetch_add((r0.elapsed().as_secs_f64() * 1e6) as u64, Ordering::Relaxed);
                                let job = if zones != -1 {
                                    stats.recoded.fetch_add(1, Ordering::Relaxed);
                                    let rows = (0..(oh + 15) / 16).filter(|&y| zones_for_rows(&bands, region_mb.unwrap(), y * 16, y * 16 + 16) & zones & !1 != 0).count();
                                    stats.recoded_rows.fetch_add(rows as i64, Ordering::Relaxed);
                                    let qp = gop.plan[j].1 as f32;
                                    let rm = region_mb.unwrap();
                                    let quant = orbit_frames.as_ref().unwrap().side().qp.iter().enumerate()
                                        .map(|(mb, &q)| if q > 51 || rm.holds(mb as i32) { QUANT_OFFSET_AQ } else { q as f32 - qp })
                                        .collect();
                                    Job::Replay { n: k, gop, region: buf, zones, full: true, quant }
                                } else {
                                    Job::Live { n: k, yuv: buf, record: None }
                                };
                                queue.lock().unwrap().ready.push_back(job);
                                changed.notify_all();
                                // Its buffers go too, five seconds after the players do; and
                                // while it draws, what only whole frames need goes now.
                                replaying_for = 0;
                                if !planner.rendering {
                                    renderer.keep_overlay_only();
                                    // Once a group, what the players left behind goes.
                                    if j == 0 {
                                        renderer.release_overlay_pages();
                                    }
                                }
                                k += 1;
                                continue;
                            }
                            Err(e) => {
                                // Rendered instead; after a few of these, for good.
                                overlay_faults += 1;
                                eprintln!("overlay: {e}, rendering instead{}", if overlay_faults >= 10 { " from now on" } else { "" });
                                overlay_on = overlay_faults < 10;
                                orbit_frames = None;
                                recode = None;
                                planner.begin(k);
                                renderer.restart();
                            }
                        }
                    }
                    match plan {
                        Ok(gop) => {
                            // Replaying with nothing drawn for a while: the renderer's buffers go,
                            // and the decoder overlay mode draws over.
                            if planner.rendering || verify.is_some() {
                                replaying_for = 0;
                            } else {
                                replaying_for += 1;
                                if replaying_for == 5 * fps {
                                    renderer.release();
                                    orbit_frames = None;
                                    recode = None;
                                    crate::sys::trim_memory();
                                }
                            }
                            if planner.rendering || verify.is_some() {
                                scene.draw(renderer, entities, &pose, at, at + engine_offset);
                                scene.latencies.clear();
                                planner.drawn(renderer, &players, k);
                                stats.background.fetch_add(1, Ordering::Relaxed);
                                if let Some(v) = &mut verify {
                                    v.check(t.slot(k), renderer.output(), ow as usize, oh as usize);
                                }
                            }
                            let mut yuv = Vec::new();
                            if let (Some(r), Some(o)) = (&mut region, overlay.as_deref_mut()) {
                                yuv.resize(r.bytes(), 0);
                                if let Err(e) = r.store.read(t.slot(k), &mut yuv) {
                                    eprintln!("replay: cannot read the region store: {e}");
                                }
                                from_yuv420(&yuv, r.w, r.h, &mut r.rgb);
                                if let Some(drawn) = o.draw_playing(&mut r.rgb, (r.x as i32, r.y as i32), r.w as i32, r.h as i32, at) {
                                    r.yuv.resize(r.bytes(), 0);
                                    to_yuv420(&r.rgb, &mut r.yuv, r.w as i32, r.h as i32, jobs);
                                    // Only where the card is drawn: elsewhere the scene is kept
                                    // as recorded, rather than through RGB and back.
                                    copy_rect(&r.yuv, &mut yuv, r.w, r.h, drawn);
                                }
                            }
                            stats.replayed.fetch_add(1, Ordering::Relaxed);
                            Job::Replay { n: k, gop, region: yuv, zones: 1, full: false, quant: Vec::new() }
                        }
                        Err(why) => {
                            let r0 = Instant::now();
                            let drawn_at = engine_clock();
                            scene.draw(renderer, entities, &pose, at, at + engine_offset);
                            planner.drawn(renderer, &players, k);
                            stats.smooth(r0.elapsed().as_secs_f64() * 1000.0);
                            stats.rendered.fetch_add(1, Ordering::Relaxed);
                            stats.why[why as usize].fetch_add(1, Ordering::Relaxed);
                            for late in scene.latencies.drain(..) {
                                let ms = (late * 1000.0).max(0.0) as u64;
                                stats.edits[0].fetch_add(1, Ordering::Relaxed);
                                stats.edits[1].fetch_add(ms, Ordering::Relaxed);
                                stats.edits[2].fetch_max(ms, Ordering::Relaxed);
                            }
                            let mut record = (replay_on && planner.recordable(k))
                                .then(|| planner.view.clone().map(|view| Rec { view, eye: pose.eye, at: drawn_at }))
                                .flatten();
                            if let (Some(r), Some(_)) = (&mut region, &record) {
                                // The scene under the card is stored per slot, so a recording
                                // of this slot from another camera path no longer matches it.
                                let (g, j) = t.place(k);
                                cache.check(g, |gop| same(gop.eyes[j], pose.eye));
                                r.take(renderer.output(), ow as usize, jobs);
                                if let Some(v) = &mut verify {
                                    v.thumbs[t.slot(k)] = Some(Verify::thumb(renderer.output(), ow as usize, oh as usize));
                                }
                                if let Err(e) = r.store.write(t.slot(k), &r.yuv) {
                                    eprintln!("replay: cannot write the region store: {e}");
                                    record = None;
                                }
                            }
                            let frame = renderer.output_mut();
                            if let Some(o) = overlay.as_deref_mut() {
                                o.draw(frame, at);
                            }
                            buf.resize(frame_bytes, 0);
                            to_yuv420(frame, &mut buf, ow, oh, jobs);
                            Job::Live { n: k, yuv: buf, record }
                        }
                    }
                } else {
                    still.clone_from(&waiting);
                    if let Some(o) = overlay.as_deref_mut() {
                        o.draw(&mut still, at);
                    }
                    buf.resize(frame_bytes, 0);
                    to_yuv420(&still, &mut buf, ow, oh, jobs);
                    Job::Live { n: k, yuv: buf, record: None }
                };
                queue.lock().unwrap().ready.push_back(job);
                changed.notify_all();
                k += 1;
                if let Some(v) = &mut verify
                    && verify_log.elapsed().as_secs() >= 60
                {
                    v.report();
                    verify_log = Instant::now();
                }
            }
        });

        // The encoder.
        s.spawn(|| {
            // It yields to the renderer, as ffmpeg did at nice 5: it has frames buffered and
            // catches up in the renderer's idle moments, while a renderer that loses a core
            // misses frames. x264's threads inherit this.
            sys::lower_priority(5);
            let mut video = match Video::new(cfg, t, region_mb, cache.clone(), tx) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("rfengine: encoder: {e}");
                    sys::QUIT.store(true, Ordering::SeqCst);
                    changed.notify_all();
                    return;
                }
            };
            let start = Instant::now() + Duration::from_secs_f64((epoch - unix_now()).max(0.0));
            std::thread::sleep(start.saturating_duration_since(Instant::now()));
            if !epoch_file.is_empty() {
                // Frame n is shown at epoch + n / fps. The radio paces its samples against the
                // same instant, so the music and the now playing card line up in ffmpeg.
                let tmp = format!("{epoch_file}.tmp");
                if std::fs::write(&tmp, format!("{epoch:.6}\n")).is_ok() {
                    let _ = std::fs::rename(&tmp, epoch_file);
                }
            }
            let mut last = vec![0u8; frame_bytes];
            let mut have_last = false;
            let mut last_replayed: Option<i64> = None;
            let mut n: i64 = 0;
            while !quitting() {
                let due = start + Duration::from_secs_f64(n as f64 * dt);
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
                let job = {
                    let mut q = queue.lock().unwrap();
                    loop {
                        // Frames made for slots already passed were superseded by repeats.
                        while q.ready.front().is_some_and(|j| j.n() < n) {
                            if let Some(Job::Live { yuv, .. }) = q.ready.pop_front() {
                                q.spare.push(yuv);
                            }
                        }
                        if q.ready.front().is_some() {
                            break q.ready.pop_front();
                        }
                        // A rendered frame late is shown again; a replayed one is always
                        // quick, and after one the next frame is waited for, as a
                        // replayed picture cannot be repeated.
                        if have_last && last_replayed.is_none() || quitting() {
                            break None;
                        }
                        q = changed.wait_timeout(q, Duration::from_millis(20)).unwrap().0;
                    }
                };
                let e0 = Instant::now();
                let result = match job {
                    Some(Job::Live { yuv, record, .. }) => {
                        let r = video.live(n, &yuv, record);
                        queue.lock().unwrap().spare.push(std::mem::replace(&mut last, yuv));
                        have_last = true;
                        last_replayed = None;
                        r
                    }
                    Some(Job::Replay { gop, region, zones, full, quant, .. }) if t.place(n).1 == 0 || last_replayed == Some(n - 1) => {
                        last_replayed = Some(n);
                        let r = video.replay(n, gop, &region, zones, full, &quant);
                        if full {
                            queue.lock().unwrap().spare.push(region);
                        }
                        // A replay that goes wrong (a recording unreadable, the recode off the
                        // recorded frame types) is given up: the last rendered frame goes out
                        // again and the render thread takes over, as when a replay is refused.
                        match r {
                            Err(e) if have_last => {
                                eprintln!("replay: {e}; rendering from here");
                                video.abandon_replay();
                                queue.lock().unwrap().refused = true;
                                last_replayed = None;
                                stats.repeats.fetch_add(1, Ordering::Relaxed);
                                video.live(n, &last, None)
                            }
                            r => r,
                        }
                    }
                    // Missing, or a replay that cannot start mid-group: the last rendered
                    // frame again, and a replay refused is rendered from here on.
                    refused => {
                        if matches!(refused, Some(Job::Replay { .. })) {
                            queue.lock().unwrap().refused = true;
                        }
                        if !have_last {
                            n += 1;
                            continue;
                        }
                        stats.repeats.fetch_add(1, Ordering::Relaxed);
                        last_replayed = None;
                        video.live(n, &last, None)
                    }
                };
                stats.encode_us.fetch_add(e0.elapsed().as_micros() as u64, Ordering::Relaxed);
                stats.encoded.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = result {
                    eprintln!("rfengine: encoder: {e}");
                    sys::QUIT.store(true, Ordering::SeqCst);
                    break;
                }
                queue.lock().unwrap().writer_index = n;
                changed.notify_all();
                n += 1;
            }
            changed.notify_all();
        });

        write(rx, cfg.fps as f64, &stats, feed, &cache, t.groups());
        sys::QUIT.store(true, Ordering::SeqCst);
        changed.notify_all();
    });
}

/// Puts the last run's recordings in the cache once the world worker has compared the world
/// with theirs; the changes it found then throw away those that could show them.
fn adopt_kept(scene: &Scene, cache: &Cache, kept: &mut Vec<(usize, Gop)>) {
    if kept.is_empty() {
        return;
    }
    let Some(from) = *scene.keep.ready_at.lock().unwrap() else { return };
    for (g, mut gop) in kept.drain(..) {
        if cache.get(g).is_none() {
            gop.from = from;
            cache.put_loaded(g, gop);
        }
    }
}

/// Everything recordings of the orbit depend on besides the world: the picture's settings,
/// the encoder's parameter sets, the overlay's static cards, and the engine build that
/// rendered them. None if it cannot be worked out, and then nothing is kept.
fn fingerprint(cfg: &Config, region: Option<Region>, gop: i64) -> Option<u64> {
    // RF_KEEP_ANY_BUILD=1 keeps recordings across builds: for testing changes that cannot
    // alter what is rendered, never for the stream.
    let exe = if std::env::var_os("RF_KEEP_ANY_BUILD").is_some() { Vec::new() } else { std::fs::read("/proc/self/exe").ok()? };
    let settings = x264_settings(cfg, region, gop);
    let (sps, pps) = Encoder::open(&settings).and_then(|mut e| e.parameter_sets()).ok()?;
    // The encoder's settings in full: bands, say, change no parameter set, but recordings
    // made without them cannot have bands recoded.
    // The overlay's text is in every recorded picture, so it counts as well.
    let text = format!(
        "{settings:?} {} {} {} {} {:?} {} {} {} {} {} {} {} {:?} {} {} {:?} {} {} {:?} {}",
        cfg.width, cfg.height, cfg.fps, cfg.period, cfg.center, cfg.radius, cfg.orbit_height, cfg.look_height, cfg.fov,
        cfg.fancy_leaves_radius, cfg.static_every, cfg.sway, cfg.look, cfg.bitrate, cfg.preset, region, cfg.assets, cfg.fonts,
        cfg.text, crate::persist::hash(&exe, 3)
    );
    let mut bytes = text.into_bytes();
    bytes.extend_from_slice(&sps);
    bytes.extend_from_slice(&pps);
    Some(crate::persist::hash(&bytes, 4))
}

const KEEP_MAGIC: u64 = u64::from_le_bytes(*b"RFKEEP01");

/// The file a run keeps: the world snapshot, then the recordings' index (replay.rs).
fn read_keep(bytes: &[u8], fp: u64, cache: &Cache) -> Result<(HashMap<u64, u64>, Vec<(usize, Gop)>), String> {
    let mut r = crate::persist::Reader::new(bytes);
    if r.u64()? != KEEP_MAGIC || r.u64()? != fp {
        return Err("recorded with other settings or another build".into());
    }
    let n = r.len(16)?;
    let mut world = HashMap::with_capacity(n);
    for _ in 0..n {
        world.insert(r.u64()?, r.u64()?);
    }
    // Magic, fingerprint and count: 20 bytes before the pairs.
    Ok((world, cache.read_index(&bytes[20 + 16 * n..], fp, 0.0)?))
}

/// Writes what this run keeps, at most once a minute, off the render thread.
struct Keeper {
    path: std::path::PathBuf,
    fingerprint: Option<u64>,
    written: u64, // a hash of what was last written
    writing: Option<std::thread::JoinHandle<()>>,
}

impl Keeper {
    fn checkpoint(&mut self, world: Vec<(u64, u64)>, cache: &Cache) {
        let Some(fp) = self.fingerprint else { return };
        if self.writing.as_ref().is_some_and(|t| !t.is_finished()) {
            return;
        }
        let index = cache.index(fp);
        let mut w = crate::persist::Writer::default();
        w.u64(KEEP_MAGIC);
        w.u64(fp);
        w.len(world.len());
        for (k, h) in &world {
            w.u64(*k);
            w.u64(*h);
        }
        w.bytes.extend_from_slice(&index);
        let sum = crate::persist::hash(&w.bytes, 5);
        if sum == self.written {
            return;
        }
        self.written = sum;
        let path = self.path.clone();
        self.writing = Some(std::thread::spawn(move || {
            if let Err(e) = crate::persist::write_atomic(&path, &w.bytes) {
                eprintln!("replay: cannot keep the recordings for the next run: {e}");
            }
        }));
    }
}

/// Renders and records one whole orbit before the stream starts, as fast as the box
/// allows rather than at the frame rate: for measuring replay on a machine too slow to
/// record the orbit in real time, and for a box that should replay from its first minute.
/// Nothing goes out meanwhile.
#[allow(clippy::too_many_arguments)]
fn prerecord_orbit(cfg: &Config, scene: &mut Scene, renderer: &mut Renderer, entities: &mut Entities, mut overlay: Option<&mut Overlay>,
                   jobs: &Jobs, t: Timeline, epoch: f64, mut region: Option<&mut RegionPixels>, region_mb: Option<Region>, cache: &Arc<Cache>) {
    while !scene.ready() {
        if quitting() {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let (ow, oh) = (renderer.width(), renderer.height());
    let (tx, rx) = sync_channel::<Out>(64);
    let drain = std::thread::spawn(move || while rx.recv().is_ok() {});
    let mut video = match Video::new(cfg, t, region_mb, cache.clone(), tx) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("prerecord: {e}");
            return;
        }
    };
    let dt = 1.0 / cfg.fps as f64;
    let engine_offset = engine_clock() - unix_now();
    // The first frame of the stream's slot 0, and a few seconds before it to settle the
    // renderer's history.
    let n0 = (-t.base).rem_euclid(t.orbit);
    let warm = 4 * cfg.fps as i64;
    let started = Instant::now();
    let mut view: Option<Arc<View>> = None;
    let mut yuv = vec![0u8; (ow * oh * 3 / 2) as usize];
    // Only groups not already recorded (kept from the last run), each from a few seconds
    // before it unless the group before it was just drawn.
    let mut frames = Vec::new();
    for g in 0..t.groups() as i64 {
        let start = n0 + g * t.gop;
        if cache.get(t.place(start).0).is_some() {
            continue;
        }
        let from = if frames.last() == Some(&(start - 1)) { start } else { start - warm };
        frames.extend(from..start + t.gop);
    }
    let mut last = i64::MIN;
    for n in frames {
        if quitting() {
            break;
        }
        if n != last + 1 {
            renderer.restart();
        }
        last = n;
        let at = epoch + n as f64 * dt;
        let (pose, _) = scene.prepare(renderer, entities, at, at + engine_offset);
        let drawn_at = engine_clock();
        scene.draw(renderer, entities, &pose, at, at + engine_offset);
        scene.latencies.clear();
        if renderer.stats.full {
            view = Some(Arc::new(renderer.sight()));
        }
        let fresh = n >= n0 && cache.get(t.place(n).0).is_none();
        let record = fresh.then(|| view.clone().map(|view| Rec { view, eye: pose.eye, at: drawn_at })).flatten();
        if let (Some(r), Some(_)) = (region.as_deref_mut(), &record) {
            r.take(renderer.output(), ow as usize, jobs);
            if let Err(e) = r.store.write(t.slot(n), &r.yuv) {
                eprintln!("prerecord: cannot write the region store: {e}");
            }
        }
        let frame = renderer.output_mut();
        if let Some(o) = overlay.as_deref_mut() {
            o.draw(frame, at);
        }
        to_yuv420(frame, &mut yuv, ow, oh, jobs);
        if let Err(e) = video.live(n, &yuv, record) {
            eprintln!("prerecord: {e}");
            break;
        }
        if (n - n0) % (60 * cfg.fps as i64) == 0 && n >= n0 {
            eprintln!("prerecord: {} of {} frames, {} groups recorded, {:.0} s", n - n0, t.orbit, cache.recorded(),
                      started.elapsed().as_secs_f64());
        }
    }
    let _ = video.end_live();
    drop(video);
    let _ = drain.join();
    renderer.restart();
    eprintln!("prerecord: {} of {} groups recorded in {:.0} s", cache.recorded(), t.groups(), started.elapsed().as_secs_f64());
}

/// The writer: FLV on stdout, each picture sent `delay` frames after its slot, and the
/// minute's log line.
fn write(rx: Receiver<Out>, fps: f64, stats: &Stats, feed: &FeedServer, cache: &Cache, groups: usize) {
    let out = match sys::stdout_file() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("rfengine: stdout: {e}");
            return;
        }
    };
    sys::grow_pipe(&out, 1 << 20);
    let mut flv = None;
    let (mut start, mut delay) = (Instant::now(), 0i64);
    let (mut blocked_ms, mut blocked_max, mut bytes) = (0.0f64, 0.0f64, 0usize);
    let mut next_log = Instant::now() + Duration::from_secs(60);
    while let Ok(msg) = rx.recv() {
        match msg {
            Out::Start { sps, pps, delay: d } => {
                let mut f = match Flv::new(std::io::BufWriter::with_capacity(1 << 20, &out)) {
                    Ok(f) => f,
                    Err(_) => return,
                };
                if f.sequence_header(&sps, &pps).is_err() {
                    return;
                }
                flv = Some(f);
                delay = d;
                start = Instant::now();
            }
            Out::Picture { d, pts, key, data } => {
                let Some(f) = flv.as_mut() else { continue };
                if d == 0 {
                    // The first picture fixes the clock the rest keep to.
                    start = Instant::now();
                }
                let release = start + Duration::from_secs_f64((d + delay) as f64 / fps);
                match release.checked_duration_since(Instant::now()) {
                    Some(wait) => std::thread::sleep(wait),
                    None => {
                        let late = (Instant::now() - release).as_secs_f64() * 1000.0;
                        stats.late.fetch_add(1, Ordering::Relaxed);
                        stats.late_max_ms.fetch_max(late as u64, Ordering::Relaxed);
                    }
                }
                // Timestamps in milliseconds: decoding order by picture count, display by
                // frame, two later for the B-frames that go out after the frame after them.
                let ms = |frames: i64| (frames as f64 * 1000.0 / fps).round() as i64;
                let (dts, pts) = (ms(d), ms(pts + 2));
                let w0 = Instant::now();
                if f.picture(dts, (pts - dts) as i32, key, &data).is_err() {
                    return;
                }
                let blocked = w0.elapsed().as_secs_f64() * 1000.0;
                blocked_ms += blocked;
                blocked_max = blocked_max.max(blocked);
                bytes += data.len();
            }
        }
        if Instant::now() >= next_log {
            next_log += Duration::from_secs(60);
            let shown = stats.edits[0].swap(0, Ordering::Relaxed);
            let (sum, longest) = (stats.edits[1].swap(0, Ordering::Relaxed), stats.edits[2].swap(0, Ordering::Relaxed));
            let why = stats.why.each_ref().map(|w| w.swap(0, Ordering::Relaxed));
            let overlaid = stats.overlaid.swap(0, Ordering::Relaxed);
            let overlay_ms = stats.overlay_ms.swap(0, Ordering::Relaxed) as f64 / 1000.0 / overlaid.max(1) as f64;
            let recoded = stats.recoded.swap(0, Ordering::Relaxed);
            let recoded_rows = stats.recoded_rows.swap(0, Ordering::Relaxed) as f64 / recoded.max(1) as f64;
            let encode_ms = stats.encode_us.swap(0, Ordering::Relaxed) as f64 / 1000.0 / stats.encoded.swap(0, Ordering::Relaxed).max(1) as f64;
            eprintln!(
                "stream: {} rendered ({:.1} ms avg; {} players in view, {} not recorded, {} camera moved, {} mid-group), {} in the background, {} replayed ({} of {groups} groups recorded), {overlaid} with players drawn over ({overlay_ms:.1} ms avg, {recoded} of them in bands, {recoded_rows:.1} rows of them, {} clipped), encoder {encode_ms:.1} ms a frame, {} repeated, {} sent late (max {} ms), {:.1} Mbit/s, ffmpeg backpressure {:.0} ms (max {:.0}), feed {}{}",
                stats.rendered.swap(0, Ordering::Relaxed),
                f64::from_bits(stats.render_ms.load(Ordering::Relaxed)),
                why[Why::Players as usize],
                why[Why::Unrecorded as usize],
                why[Why::Moved as usize],
                why[Why::Midway as usize],
                stats.background.swap(0, Ordering::Relaxed),
                stats.replayed.swap(0, Ordering::Relaxed),
                cache.recorded(),
                stats.clipped.swap(0, Ordering::Relaxed),
                stats.repeats.swap(0, Ordering::Relaxed),
                stats.late.swap(0, Ordering::Relaxed),
                stats.late_max_ms.swap(0, Ordering::Relaxed),
                bytes as f64 * 8.0 / 60e6,
                blocked_ms,
                blocked_max,
                if feed.connected() { "connected" } else { "down" },
                if shown > 0 { format!(", {shown} edits out {} ms after arriving (longest {longest})", sum / shown) } else { String::new() }
            );
            (blocked_ms, blocked_max, bytes) = (0.0, 0.0, 0);
        }
    }
}
