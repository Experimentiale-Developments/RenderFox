//! Receives the observer plugin's feed (docs/protocol.md) and applies it.
//!
//! Only one game server feeds the engine: the one it is paired with, whose Ed25519 public
//! key is the config's observerKey or, more usually, the key file pairing wrote. Each
//! connection gets a fresh random nonce, and the observer's hello must carry that key and
//! a signature over the nonce, so a recorded handshake cannot be replayed and a copy of
//! the plugin on another server is refused.
//!
//! An engine with no key is unpaired: it makes a one-time code, and the first server to
//! prove it knows the code (an HMAC over the nonce and its key) and holds its key's
//! private half becomes the paired one. Five wrong codes and the code is replaced.

use crate::assets::Assets;
use crate::base64;
use crate::vmath::Vec3;
use crate::world::World;
use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Seconds since the engine started, on a monotonic clock.
pub fn engine_clock() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EntitySample {
    pub time: f64, // engine clock, seconds, of the server tick it was taken on
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub body_yaw: f32,
    pub pose: u8, // org.bukkit.entity.Pose ordinal; sneaking comes from the flags
    pub flags: u8,
}

#[derive(Clone, Debug)]
pub struct EntityState {
    pub id: i32,
    pub kind: String,
    pub name: String,
    pub skin: String,
    pub gear: Vec<String>, // main, off, head, chest, legs, feet
    pub samples: VecDeque<EntitySample>,
    pub last_swing: f64,
    pub last_swing_off: f64,
    pub last_hurt: f64,
    pub last_seen: f64,
}

impl EntityState {
    fn new(id: i32) -> EntityState {
        EntityState {
            id,
            kind: String::new(),
            name: String::new(),
            skin: String::new(),
            gear: Vec::new(),
            samples: VecDeque::new(),
            last_swing: -10.0,
            last_swing_off: -10.0,
            last_hurt: -10.0,
            last_seen: 0.0,
        }
    }
}

pub const EF_SNEAKING: u8 = 1;
pub const POSE_FALL_FLYING: u8 = 1;
pub const EF_SWING_MAIN: u8 = 4;
pub const EF_SWING_OFF: u8 = 8;
pub const EF_HURT: u8 = 16;

#[derive(Clone, Debug)]
pub struct Hello {
    pub world: String,
    pub min_y: i32,
    pub max_y: i32,
    pub center_x: i32,
    pub center_z: i32,
    pub radius: i32,
    #[allow(dead_code)] // on the wire; interpolation adapts to sample times instead
    pub entity_hz: i32,
}

const HELLO: u8 = 1;
const SECTIONS: u8 = 2;
const SNAPSHOT_DONE: u8 = 3;
const ENTITIES: u8 = 4;
const ENTITY_INFO: u8 = 5;
const PING: u8 = 6;
const CHALLENGE: u8 = 7;
const ACCEPT: u8 = 8;
const PAIR: u8 = 9;
const PAIRED: u8 = 10;
const REFUSED: u8 = 11;
const PROTOCOL: i32 = 2;
const HELLO_CONTEXT: &[u8] = b"RenderFox observer hello v2";
const PAIR_CONTEXT: &[u8] = b"RenderFox pair v1";
// Crockford's base32: no I, L, O or U, so a code read aloud or off a screen survives.
const CODE_ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const CODE_LENGTH: usize = 10;
const PAIR_TRIES: u32 = 5;
// Samples kept per entity: two seconds at twenty a second, past any posing delay.
const SAMPLES: usize = 40;

/// Big-endian reader over a byte buffer, matching java.io.DataOutputStream.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

type ReadResult<T> = Result<T, String>;

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> ReadResult<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return Err("truncated message".into());
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> ReadResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> ReadResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> ReadResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> ReadResult<i32> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> ReadResult<i64> {
        let hi = self.u32()? as u64;
        Ok((hi << 32 | self.u32()? as u64) as i64)
    }
    fn f32(&mut self) -> ReadResult<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> ReadResult<f64> {
        Ok(f64::from_bits(self.i64()? as u64))
    }
    fn utf(&mut self) -> ReadResult<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn long_utf(&mut self) -> ReadResult<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

pub type OnHello = Box<dyn Fn(&Hello) -> Arc<World> + Send + Sync>;

/// Which server may feed the engine, and where pairing keeps its state.
pub struct Trust {
    /// observerKey from the config, or the key file's; None until paired.
    pub key: Option<VerifyingKey>,
    /// Where pairing writes the paired server's public key.
    pub key_file: String,
    /// Where an unpaired engine puts its code for the `renderfox` tool to show.
    pub code_file: String,
}

impl Trust {
    /// The config's key if it has one, else the key file's, else none.
    pub fn load(config_key: Option<VerifyingKey>, key_file: String, code_file: String) -> Result<Trust, String> {
        let key = match config_key {
            Some(k) => Some(k),
            None => match std::fs::read_to_string(&key_file) {
                Ok(text) => Some(parse_key(text.trim()).map_err(|e| format!("{key_file}: {e}"))?),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(format!("cannot read {key_file}: {e}")),
            },
        };
        Ok(Trust { key, key_file, code_file })
    }
}

pub fn parse_key(text: &str) -> Result<VerifyingKey, String> {
    let raw: [u8; 32] = base64::decode(text).try_into().map_err(|_| format!("{text} is not a 32-byte base64 key"))?;
    VerifyingKey::from_bytes(&raw).map_err(|_| format!("{text} is not an Ed25519 public key"))
}

struct Pairing {
    code: String,
    failures: u32,
}

struct Shared {
    assets: Arc<Assets>,
    key: Mutex<Option<VerifyingKey>>,
    pairing: Mutex<Option<Pairing>>,
    key_file: String,
    code_file: String,
    on_hello: OnHello,
    last_refusal: Mutex<(String, f64)>,
    stop: AtomicBool,
    connected: AtomicBool,
    snapshot_done: AtomicBool,
    bytes: AtomicU64,
    client: Mutex<Option<TcpStream>>,
    entities: Mutex<BTreeMap<i32, EntityState>>,
}

pub struct FeedServer {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl FeedServer {
    /// `on_hello` is called when a HELLO arrives and returns the world to fill.
    pub fn new(assets: Arc<Assets>, port: i32, trust: Trust, on_hello: OnHello) -> FeedServer {
        let listener = TcpListener::bind(("0.0.0.0", port as u16))
            .unwrap_or_else(|e| panic!("cannot listen on port {port}: {e}"));
        // Polled rather than blocking, so the server can be stopped.
        listener.set_nonblocking(true).expect("nonblocking listener");
        let paired = trust.key.is_some();
        let shared = Arc::new(Shared {
            assets,
            key: Mutex::new(trust.key),
            pairing: Mutex::new(None),
            key_file: trust.key_file,
            code_file: trust.code_file,
            on_hello,
            last_refusal: Mutex::new((String::new(), 0.0)),
            stop: AtomicBool::new(false),
            connected: AtomicBool::new(false),
            snapshot_done: AtomicBool::new(false),
            bytes: AtomicU64::new(0),
            client: Mutex::new(None),
            entities: Mutex::new(BTreeMap::new()),
        });
        if paired {
            let _ = std::fs::remove_file(&shared.code_file);
        } else {
            new_code(&shared, port);
        }
        let s = shared.clone();
        let thread = std::thread::Builder::new()
            .name("feed".into())
            .spawn(move || accept_loop(&s, listener))
            .expect("spawn feed thread");
        FeedServer { shared, thread: Some(thread) }
    }

    pub fn connected(&self) -> bool {
        self.shared.connected.load(Ordering::SeqCst)
    }

    pub fn snapshot_done(&self) -> bool {
        self.shared.snapshot_done.load(Ordering::SeqCst)
    }

    /// Every entity with a known type and at least one sample.
    pub fn entities(&self, _now: f64) -> Vec<EntityState> {
        let g = self.shared.entities.lock().unwrap();
        g.values().filter(|e| !e.samples.is_empty() && !e.kind.is_empty()).cloned().collect()
    }
}

impl Drop for FeedServer {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(c) = self.shared.client.lock().unwrap().as_ref() {
            let _ = c.shutdown(Shutdown::Both);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn accept_loop(shared: &Shared, listener: TcpListener) {
    while !shared.stop.load(Ordering::SeqCst) {
        let (stream, peer) = match listener.accept() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(_) => continue,
        };
        let ip = peer.ip().to_string();
        let _ = stream.set_nonblocking(false);
        *shared.client.lock().unwrap() = stream.try_clone().ok();
        serve(shared, stream, &ip);
        *shared.client.lock().unwrap() = None;
        if shared.connected.swap(false, Ordering::SeqCst) {
            eprintln!("feed: {ip} disconnected");
        }
    }
}

fn serve(shared: &Shared, mut stream: TcpStream, peer: &str) {
    let Some(world) = handshake(shared, &mut stream, peer) else { return };
    // A dead peer must not hold the only feed slot forever; the plugin pings every
    // five seconds, so thirty seconds of silence means it is gone.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut clock = TickClock::default();
    let mut payload: Vec<u8> = Vec::new();
    while let Some(kind) = read_frame(shared, &mut stream, &mut payload, 64 << 20) {
        if kind == HELLO {
            eprintln!("feed: {peer} sent a second hello, closing");
            return;
        }
        if let Err(e) = handle(shared, &world, &mut clock, kind, &payload) {
            eprintln!("feed: bad message type {kind}: {e}");
            return;
        }
    }
}

/// One message's type, with its payload in `payload`; None when the peer is gone or
/// the frame is not believable.
fn read_frame(shared: &Shared, stream: &mut TcpStream, payload: &mut Vec<u8>, max: u32) -> Option<u8> {
    let mut head = [0u8; 5];
    stream.read_exact(&mut head).ok()?;
    let length = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    if !(1..=max).contains(&length) {
        eprintln!("feed: bad frame length {length}");
        return None;
    }
    payload.resize(length as usize - 1, 0);
    stream.read_exact(payload).ok()?;
    shared.bytes.fetch_add(length as u64 + 4, Ordering::SeqCst);
    Some(head[4])
}

/// The world to fill, once the peer has signed this connection's nonce with the
/// configured key; None when it is refused or goes away.
fn handshake(shared: &Shared, stream: &mut TcpStream, peer: &str) -> Option<Arc<World>> {
    let mut nonce = [0u8; 32];
    if let Err(e) = File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut nonce)) {
        eprintln!("feed: no randomness for a challenge: {e}");
        return None;
    }
    let mut challenge = Vec::with_capacity(41);
    challenge.extend_from_slice(&37u32.to_be_bytes());
    challenge.push(CHALLENGE);
    challenge.extend_from_slice(&PROTOCOL.to_be_bytes());
    challenge.extend_from_slice(&nonce);
    // An observer answers at once; whatever holds the slot silently is dropped.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    stream.write_all(&challenge).ok()?;
    let mut payload = Vec::new();
    let kind = read_frame(shared, stream, &mut payload, 64 << 10)?;
    if kind == PAIR {
        let answer = match pair(shared, &nonce, &payload, peer) {
            Ok(()) => frame(PAIRED, &[]),
            Err(why) => {
                refuse(shared, peer, &why);
                let mut reason = Vec::with_capacity(why.len() + 2);
                let why = &why.as_bytes()[..why.len().min(1000)];
                reason.extend_from_slice(&(why.len() as u16).to_be_bytes());
                reason.extend_from_slice(why);
                frame(REFUSED, &reason)
            }
        };
        let _ = stream.write_all(&answer);
        return None;
    }
    let hello = if kind == HELLO { check_hello(shared, &nonce, &payload) } else { Err("did not start with a hello".into()) };
    let hello = match hello {
        Ok(h) => h,
        Err(why) => {
            refuse(shared, peer, &why);
            return None;
        }
    };
    stream.write_all(&[0, 0, 0, 1, ACCEPT]).ok()?;
    let w = (shared.on_hello)(&hello);
    shared.connected.store(true, Ordering::SeqCst);
    shared.snapshot_done.store(false, Ordering::SeqCst);
    eprintln!("feed: hello from {peer} with the configured key, {}, area {},{} r{}", hello.world, hello.center_x, hello.center_z,
              hello.radius);
    Some(w)
}

fn check_hello(shared: &Shared, nonce: &[u8; 32], payload: &[u8]) -> Result<Hello, String> {
    let protocol = Reader::new(payload).i32()?;
    if protocol != PROTOCOL {
        return Err(format!("speaks protocol {protocol} and this engine {PROTOCOL}: update the observer plugin"));
    }
    let Some(trusted) = *shared.key.lock().unwrap() else {
        return Err("said hello, but this engine is not paired yet: pair it with /renderfox pair on the game server".into());
    };
    let (body, signature) = payload.split_at(payload.len().checked_sub(64).ok_or("sent a truncated hello")?);
    let mut r = Reader::new(body);
    r.i32()?;
    let key: [u8; 32] = r.take(32)?.try_into().unwrap();
    if key != trusted.to_bytes() {
        return Err(format!("is a server whose key is {}, not the one this engine is paired with", base64::encode(&key)));
    }
    let mut signed = Vec::with_capacity(HELLO_CONTEXT.len() + nonce.len() + body.len());
    signed.extend_from_slice(HELLO_CONTEXT);
    signed.extend_from_slice(nonce);
    signed.extend_from_slice(body);
    let signature = Signature::from_bytes(signature.try_into().unwrap());
    trusted.verify_strict(&signed, &signature).map_err(|_| "sent a hello its key did not sign")?;
    Ok(Hello {
        world: r.utf()?,
        min_y: r.i32()?,
        max_y: r.i32()?,
        center_x: r.i32()?,
        center_z: r.i32()?,
        radius: r.i32()?,
        entity_hz: r.i32()?,
    })
}

/// PAIR: `int32 protocol, key[32], proof[32], signature[64]`, the proof an HMAC-SHA256
/// keyed with the code over the context, the nonce and the key, the signature the key's
/// over all of that and the proof. Pairing again with the key already paired succeeds,
/// so a plugin that missed the answer can simply retry.
fn pair(shared: &Shared, nonce: &[u8; 32], payload: &[u8], peer: &str) -> Result<(), String> {
    let mut r = Reader::new(payload);
    let protocol = r.i32()?;
    if protocol != PROTOCOL {
        return Err(format!("speaks protocol {protocol} and this engine {PROTOCOL}: update whichever is older"));
    }
    let key: [u8; 32] = r.take(32)?.try_into().unwrap();
    let proof: [u8; 32] = r.take(32)?.try_into().unwrap();
    let signature = Signature::from_bytes(r.take(64)?.try_into().unwrap());
    let offered = VerifyingKey::from_bytes(&key).map_err(|_| "offered a key that is not an Ed25519 public key")?;
    offered
        .verify_strict(&[PAIR_CONTEXT, nonce, &key, &proof].concat(), &signature)
        .map_err(|_| "sent a pairing request its key did not sign")?;
    let mut current = shared.key.lock().unwrap();
    if let Some(k) = *current {
        if k.to_bytes() == key {
            return Ok(());
        }
        return Err("tried to pair, but this engine is already paired with another server: run `renderfox unpair` on the render box first".into());
    }
    let mut pairing = shared.pairing.lock().unwrap();
    let Some(p) = pairing.as_mut() else { return Err("tried to pair, but this engine has no pairing code".into()) };
    if !same(&pair_proof(&p.code, nonce, &key), &proof) {
        p.failures += 1;
        if p.failures >= PAIR_TRIES {
            drop(pairing);
            eprintln!("feed: {PAIR_TRIES} wrong pairing codes, making a new one");
            new_code(shared, 0);
        }
        return Err("tried to pair with the wrong code".into());
    }
    write_key_file(&shared.key_file, &base64::encode(&key)).map_err(|e| format!("paired, but {} cannot be written: {e}", shared.key_file))?;
    *current = Some(offered);
    *pairing = None;
    let _ = std::fs::remove_file(&shared.code_file);
    eprintln!("feed: paired with {peer}, whose key is {}; saved to {}", base64::encode(&key), shared.key_file);
    Ok(())
}

/// The proof both sides compute; the code goes in as the user typed it, normalised.
pub fn pair_proof(code: &str, nonce: &[u8], key: &[u8]) -> [u8; 32] {
    hmac_sha256(normalize_code(code).as_bytes(), &[PAIR_CONTEXT, nonce, key])
}

/// Upper case without separators, with the letters Crockford's base32 reads as digits.
pub fn normalize_code(code: &str) -> String {
    code.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| match c.to_ascii_uppercase() {
            'O' => '0',
            'I' | 'L' => '1',
            c => c,
        })
        .collect()
}

fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    for part in parts {
        inner.update(part);
    }
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    outer.finalize().into()
}

fn same(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A fresh code, logged and written to the code file. `port` 0 keeps the log line short.
fn new_code(shared: &Shared, port: i32) {
    let mut random = [0u8; CODE_LENGTH];
    if let Err(e) = File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut random)) {
        eprintln!("feed: no randomness for a pairing code, pairing is off: {e}");
        return;
    }
    let raw: String = random.iter().map(|b| CODE_ALPHABET[(b & 31) as usize] as char).collect();
    let code = format!("{}-{}", &raw[..5], &raw[5..]);
    if let Err(e) = write_private(&shared.code_file, &format!("{code}\n")) {
        eprintln!("feed: cannot write {}: {e}", shared.code_file);
    }
    if port > 0 {
        eprintln!("feed: not paired with a game server yet. Pairing code {code}: on the server, run /renderfox pair <this machine>:{port} {code}");
    } else {
        eprintln!("feed: new pairing code {code}");
    }
    *shared.pairing.lock().unwrap() = Some(Pairing { code, failures: 0 });
}

fn write_private(path: &str, text: &str) -> std::io::Result<()> {
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    f.write_all(text.as_bytes())
}

/// Written beside and renamed over, so a crash never leaves half a key.
fn write_key_file(path: &str, key: &str) -> std::io::Result<()> {
    let partial = format!("{path}.partial");
    write_private(&partial, &format!("{key}\n"))?;
    std::fs::rename(&partial, path)
}

fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 5);
    out.extend_from_slice(&(payload.len() as u32 + 1).to_be_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

/// A refused server retries every few seconds; the same refusal is logged every ten
/// minutes, not every time.
fn refuse(shared: &Shared, peer: &str, why: &str) {
    let line = format!("feed: refused {peer}, which {why}");
    let now = engine_clock();
    let mut last = shared.last_refusal.lock().unwrap();
    if last.0 != line || now - last.1 > 600.0 {
        eprintln!("{line}");
        *last = (line, now);
    }
}

/// Maps the server's world tick to the engine clock. Positions are placed on the game's
/// own timeline, so a frame that arrives late behind a burst of terrain still lands
/// where it belongs instead of bunching up with the next. The offset follows the fastest
/// arrivals and creeps up when the server runs slower than twenty ticks a second.
#[derive(Default)]
struct TickClock {
    offset: Option<f64>,
    last: f64,
}

impl TickClock {
    fn at(&mut self, tick: i64, now: f64) -> f64 {
        let game = tick as f64 * 0.05;
        let seen = now - game;
        let offset = match self.offset {
            Some(o) if seen > o => o + (seen - o) * 0.05,
            _ => seen,
        };
        self.offset = Some(offset);
        let t = (game + offset).max(self.last + 1e-3).min(now);
        self.last = t;
        t
    }
}

fn handle(shared: &Shared, world: &World, clock: &mut TickClock, kind: u8, payload: &[u8]) -> ReadResult<()> {
    match kind {
        PING => {}
        SNAPSHOT_DONE => {
            shared.snapshot_done.store(true, Ordering::SeqCst);
            eprintln!(
                "feed: snapshot complete, {:.1} MiB received",
                shared.bytes.load(Ordering::SeqCst) as f64 / 1048576.0
            );
        }
        SECTIONS => {
            let plain = miniz_oxide::inflate::decompress_to_vec_zlib(payload).map_err(|e| format!("inflate failed: {e:?}"))?;
            let mut r = Reader::new(&plain);
            let (cx, cz) = (r.i32()?, r.i32()?);
            let count = r.u16()?;
            let mut ids: Vec<u16> = Vec::new();
            let mut blocks = [0u16; 4096];
            let mut biomes = [0u16; 64];
            for _ in 0..count {
                let sy = r.u8()? as i8 as i32;
                let palette = r.u16()? as usize;
                if palette == 0 {
                    world.clear_section(cx, sy, cz);
                    continue;
                }
                ids.clear();
                for _ in 0..palette {
                    ids.push(shared.assets.state_id(&r.utf()?));
                }
                if palette <= 256 {
                    let idx = r.take(4096)?;
                    for (b, &i) in blocks.iter_mut().zip(idx) {
                        *b = ids[(i as usize).min(palette - 1)];
                    }
                } else {
                    for b in blocks.iter_mut() {
                        *b = ids[(r.u16()? as usize).min(palette - 1)];
                    }
                }
                let biome_count = r.u8()? as usize;
                let mut biome_ids = [0u16; 256];
                for id in biome_ids.iter_mut().take(biome_count) {
                    *id = shared.assets.biome_id(&r.utf()?);
                }
                let cells = r.take(64)?;
                for (b, &c) in biomes.iter_mut().zip(cells) {
                    *b = biome_ids[(c as usize).min(biome_count.saturating_sub(1))];
                }
                world.set_section(cx, sy, cz, &blocks, &biomes);
            }
        }
        ENTITY_INFO => {
            let mut r = Reader::new(payload);
            let id = r.i32()?;
            let (kind, name, skin) = (r.utf()?, r.utf()?, r.long_utf()?);
            let n = r.u8()?;
            let mut gear = Vec::with_capacity(n as usize);
            for _ in 0..n {
                gear.push(r.utf()?);
            }
            let mut g = shared.entities.lock().unwrap();
            let e = g.entry(id).or_insert_with(|| EntityState::new(id));
            e.id = id;
            e.kind = kind;
            e.name = name;
            e.skin = skin;
            e.gear = gear;
        }
        ENTITIES => {
            let mut r = Reader::new(payload);
            r.i64()?;
            let tick = r.i64()?;
            let count = r.u16()?;
            let now = engine_clock();
            let time = clock.at(tick, now);
            let mut g = shared.entities.lock().unwrap();
            for _ in 0..count {
                let id = r.i32()?;
                let s = EntitySample {
                    time,
                    pos: Vec3::new(r.f64()? as f32, r.f64()? as f32, r.f64()? as f32),
                    yaw: r.f32()?,
                    pitch: r.f32()?,
                    body_yaw: r.f32()?,
                    pose: r.u8()?,
                    flags: r.u8()?,
                };
                let e = g.entry(id).or_insert_with(|| EntityState::new(id));
                e.id = id;
                e.last_seen = now;
                if s.flags & EF_SWING_MAIN != 0 {
                    e.last_swing = time;
                }
                if s.flags & EF_SWING_OFF != 0 {
                    e.last_swing_off = time;
                }
                if s.flags & EF_HURT != 0 {
                    e.last_hurt = time;
                }
                e.samples.push_back(s);
                while e.samples.len() > SAMPLES {
                    e.samples.pop_front();
                }
            }
            g.retain(|_, e| now - e.last_seen <= 1.0);
        }
        _ => eprintln!("feed: unknown message type {kind}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn hmac_matches_rfc_4231() {
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", &[b"what do ya ", b"want for nothing?"])),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha256(&[0xaa; 131], &[b"Test Using Larger Than Block-Size Key - Hash Key First"])),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    /// The same vector is in the plugin's PairingTest: both sides must agree on it.
    #[test]
    fn pair_proof_is_shared_with_the_plugin() {
        let nonce: Vec<u8> = (0..32).collect();
        let proof = pair_proof("abcde-fghjk", &nonce, &[7; 32]);
        assert_eq!(hex(&proof), "1136a1e22f523b7a87a8dd6a13808c75ef26a96e2f27c919a86f3e3a17c188d2");
    }

    #[test]
    fn codes_survive_how_people_type_them() {
        assert_eq!(normalize_code(" 7kq2m-x9prt "), "7KQ2MX9PRT");
        assert_eq!(normalize_code("O1L-Io"), "01110");
    }
}
