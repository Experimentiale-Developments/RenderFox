# Upgrading without going off air

How to swap the engine (or anything in the encoder) on a live YouTube stream without the broadcast ending or viewers seeing it drop. It uses YouTube's backup ingest, so it needs a YouTube output with the backup on (`renderfox youtube`, the default). The commands assume the checkout lives at `$RF` (for example `RF=~/renderfox`) and the new version is checked out beside it at `$NEXT`.

For most upgrades a plain `renderfox restart` is fine: YouTube shows a short drop and the broadcast resumes, as long as Auto-start is on for the stream key.

---

## 1. Architectural Foundations

### Dual-Ingest Stream Redundancy
YouTube Live supports native stream redundancy through dual ingest points:
* **Primary Ingest (`a.rtmps`):** `rtmps://a.rtmps.youtube.com:443/live2/<stream-key>`
* **Backup Ingest (`b.rtmps`):** `rtmps://b.rtmps.youtube.com:443/live2?backup=1/<stream-key>`

YouTube's ingestion cluster operates with automatic failover logic:
1. When both streams are healthy, YouTube streams video and audio from the **Primary** endpoint.
2. If Primary ceases sending data, YouTube fails over immediately (within ~1 GOP window / <2 seconds) to the **Backup** endpoint without terminating the live broadcast or disconnecting active viewers.
3. Once Primary resumes transmitting valid video conforming to ingest specifications (High profile H.264, closed GOP 48, CBR 10000k, AAC 128k 44.1 kHz stereo), YouTube seamlessly promotes Primary back to active output.

```
       +-------------------------------------------------------------+
       |                     Render Fox Pipeline                     |
       +-------------------------------------------------------------+
                       |                               |
          [Normal: renderfox-uplink]       [Normal: uplink-backup]
          [Migration: Live Primary ]       [Migration: Standby Holder]
                       |                               |
                       v                               v
             rtmps://a.rtmps.youtube...      rtmps://b.rtmps.youtube...
                       \                               /
                        \                             /
                         v                           v
                      +---------------------------------+
                      |     YouTube Live Ingestion      |
                      |  - Seamless Active/Standby      |
                      |  - Zero dropped broadcast       |
                      +---------------------------------+
```

---

## 2. Why MediaMTX Overrides Cannot Be Used For Live Cutover

MediaMTX v1.21.0 supports `overridePublisher: yes` on paths. However, **empirically verified testing** shows:
* When an overriding publisher connects to `/fm`, MediaMTX immediately disconnects all active reader clients:
  ```
  Reader DIED ON OVERRIDE: [in#0/flv] Error during demuxing: Input/output error
  ```
* Because both `renderfox-uplink` and `renderfox-uplink-backup` read from MediaMTX `/fm`, attempting to switch publishers inside MediaMTX drops both uplinks simultaneously, causing an immediate stream termination on YouTube.
* **Rule:** Never switch or override MediaMTX `/fm` directly during a live migration. Cutover must happen out-of-band using YouTube's native Backup Ingest.

---

## 3. The 4-Phase Migration Procedure

### Phase 1: Out-of-band Staging & Compilation
Build the new version in a separate checkout while the old engine keeps streaming. On a four-core box the build competes with the renderer, so expect some repeated frames while it runs, or build at low priority (`nice -n 19`).

1. **Verify / Install System Prerequisites:**
   ```bash
   sudo apt-get update && sudo apt-get install -y libavcodec-dev pkg-config nasm
   ```

2. **Check out the new version beside the running one:**
   ```bash
   git clone https://github.com/Experimentiale-Developments/RenderFox.git "$NEXT"
   git -C "$NEXT" checkout <tag or commit>
   ```

3. **Compile Patched x264:**
   ```bash
   cd "$NEXT/x264"
   bash build.sh
   # Verify static library is built:
   ls -la build/lib/libx264.a
   ```

4. **Compile New Engine Binary:**
   ```bash
   cd "$NEXT/engine"
   nice -n 19 ~/.cargo/bin/cargo build --release --locked
   ls -la target/release/rfengine
   ```

5. **Pre-populate the Audio Cache:**
   Only needed when the radio's cache format changed. Pre-encode the music into ADTS AAC in `$RF/cache/aac/` using low priority:
   ```bash
   RF="$RF" NEXT="$NEXT" python3 - << 'EOF'
   import os, sys
   sys.path.insert(0, os.environ["NEXT"] + "/stream")
   from pathlib import Path
   import radio

   cache_dir = Path(os.environ["RF"]) / "cache/aac"
   music_dir = Path(os.environ["RF"]) / "music"
   lib = radio.Library(cache_dir, None)
   for f in music_dir.glob("*.flac"):
       lib.encode(f)
   EOF
   ```

---

### Phase 2: Engage Standby Holder on YouTube Backup

1. **Pause the Watchdog Daemon:**
   ```bash
   sudo systemctl stop renderfox-watchdog
   ```

2. **Stop Backup Uplink (Primary remains 100% live):**
   ```bash
   sudo systemctl stop renderfox-uplink-backup
   ```

3. **Launch Standby Holder pushing directly to YouTube Backup Ingest:**
   ```bash
   source "$RF/relay/uplink.env"

   ffmpeg -hide_banner -loglevel warning -nostdin -re \
       -f lavfi -i "color=c=black:s=1920x1080:r=24" \
       -f lavfi -i "anullsrc=r=44100:cl=stereo" \
       -c:v libx264 -preset ultrafast -profile:v high -level 4.1 -pix_fmt yuv420p \
       -b:v 10000k -minrate 10000k -maxrate 10000k -bufsize 10000k \
       -x264-params nal-hrd=cbr:force-cfr=1 \
       -g 48 -keyint_min 48 -sc_threshold 0 -bf 2 \
       -c:a aac -b:a 128k -ar 44100 -ac 2 \
       -f flv -flvflags no_duration_filesize \
       -rw_timeout 5000000 -tcp_nodelay 1 \
       -tls_verify 1 -ca_file /etc/ssl/certs/ca-certificates.crt \
       "$BACKUP_INGEST_URL" > "$RF/run/holder.log" 2>&1 &
   HOLDER_PID=$!
   ```

4. **Verify Holder Stability:**
   Confirm with `kill -0 $HOLDER_PID` and inspect `$RF/run/holder.log`. Backup ingest is now receiving valid CBR video.

---

### Phase 3: Backend Cutover & Primary Reconnection

1. **Stop Old Engine and Primary Uplink:**
   ```bash
   sudo systemctl stop renderfox-encode renderfox-uplink
   ```
   *YouTube immediately switches stream source to Backup (Standby Holder). Viewers observe a seamless black slate with zero stream disconnection.*

2. **Install New Binary and Scripts:**
   ```bash
   cp "$RF/engine/target/release/rfengine" "$RF/run/rfengine.prev"
   cp "$NEXT/engine/target/release/rfengine" "$RF/engine/target/release/rfengine"
   cp "$NEXT/stream/encode.sh" "$RF/stream/encode.sh"
   cp "$NEXT/stream/radio.py" "$RF/stream/radio.py"
   cp "$NEXT/relay/uplink.sh" "$RF/relay/uplink.sh"
   cp "$NEXT/relay/watchdog.py" "$RF/relay/watchdog.py"
   chmod +x "$RF/stream/encode.sh" "$RF/stream/radio.py" "$RF/relay/uplink.sh" "$RF/relay/watchdog.py"
   ```

3. **Start New Encoder Service:**
   ```bash
   sudo systemctl start renderfox-encode
   ```

4. **Wait for MediaMTX `/fm` to Become Ready:**
   ```bash
   while ! curl -s http://127.0.0.1:9997/v3/paths/get/fm | grep -q '"ready":true'; do
       sleep 0.5
   done
   ```

5. **Start Primary Uplink:**
   ```bash
   sudo systemctl start renderfox-uplink
   ```
   *YouTube detects Primary back online and automatically switches output from Backup (Holder) back to Primary (Live Game Stream).*

---

### Phase 4: Disengage Holder & Restore Redundancy

1. **Terminate Standby Holder:**
   ```bash
   kill "$HOLDER_PID" 2>/dev/null || true
   wait "$HOLDER_PID" 2>/dev/null || true
   rm -f "$RF/run/holder.log"
   ```

2. **Restart Backup Uplink & Watchdog Daemon:**
   ```bash
   sudo systemctl start renderfox-uplink-backup
   sudo systemctl start renderfox-watchdog
   ```

3. **Final Health Verification:**
   ```bash
   # Both services must be active (running)
   systemctl status renderfox-* --no-pager

   # Verify MediaMTX has 2 active readers
   curl -s http://127.0.0.1:9997/v3/paths/get/fm | jq '{ready: .ready, readers: (.readers | length)}'
   ```

---

4. **Bring the checkout up to the new version**, so `$RF` matches what now runs (the files copied above are already identical, so nothing changes on disk but the commit):
   ```bash
   git -C "$RF" fetch && git -C "$RF" checkout <the same tag or commit>
   ```

## 4. Rollback Plan

If the new engine fails to publish within 30 seconds of starting:
1. Keep the Standby Holder running (the YouTube stream remains live on Backup).
2. Restore previous binary:
   ```bash
   cp "$RF/run/rfengine.prev" "$RF/engine/target/release/rfengine"
   sudo systemctl restart renderfox-encode
   ```
3. Reconnect Primary:
   ```bash
   sudo systemctl start renderfox-uplink
   ```
4. Disengage holder and restore backup uplink and watchdog as described in Phase 4.
