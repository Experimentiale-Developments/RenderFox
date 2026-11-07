# How YouTube accepts a live stream (checked 2026-09-23)

A live stream goes to YouTube as RTMP over TLS (RTMPS) or HLS.

HLS is for multi-bitrate encodes or newer codecs (HEVC, AV1). It sends small files over
HTTPS, which makes latency slightly higher and requires an HTTP server or an encoder
that pushes them.

RTMPS is YouTube's standard live ingest:
- YouTube requires **RTMPS**, not plain RTMP, since 2023 for new streams and 2024 for
  all streams; plain RTMP on 1935 is deprecated and blocked in some accounts.
- Persistent stream keys: YouTube gives a channel a persistent key that stays valid across
  broadcasts, so a box can reboot and resume without having to fetch a fresh key from the
  YouTube API each time.
- SRT exists only as a feature YouTube support enables per account; it is not open
  to a new channel. WebRTC is only for browser (webcam) streaming, not encoders.

RTMPS specifics from the Live Streaming API guide:
- connect to **port 443**, protocol `rtmps` (plain RTMP inside TLS);
- the TLS handshake **must carry SNI** set to the ingest hostname or authentication
  fails, even though the connection itself would open without it;
- the path is the RTMP application, `live2`, followed by the stream key.

Primary: `rtmps://a.rtmps.youtube.com:443/live2/<stream key>`
Backup:  `rtmps://b.rtmps.youtube.com:443/live2?backup=1/<stream key>`

Render Fox runs two concurrent uplinks from its local MediaMTX relay (`renderfox-uplink`
to Primary and `renderfox-uplink-backup` to Backup). YouTube ingests both streams
simultaneously (Stream Redundancy) and automatically switches if one drops or times out,
preventing the broadcast from terminating.

## Encoder settings (YouTube Help, "Choose live encoder settings, bitrates, and resolutions")

- Video codec H.264 (H.265/AV1 accepted over HLS/RTMP enhanced, not needed here)
- **Rate control CBR**
- **Keyframe every 2 seconds recommended, never more than 4**
- Progressive scan, B-frames 2, CABAC (so High or Main profile, not Baseline)
- Up to 60 fps
- 720p30: recommended 4 Mbps for H.264 (the 240p to 720p at 30 fps row)
- 1080p30: recommended 10 Mbps for H.264
- **Audio AAC (or MP3), stereo, 128 kbps, 44.1 kHz** (48 kHz is listed for 5.1)

Render Fox sends 1920x1080, 24 fps, H.264 High, CBR 10000 kbps (the 1080p
figure; 24 fps sits inside "24 to 60"), GOP 48 (2 s, closed, no scene-cut
keyframes), 2 B-frames, AAC-LC stereo 128 kbps 44.1 kHz. The video is encoded inside
the engine by a patched x264 (`x264/`), which runs `superfast` with `veryfast`'s mb-tree
put back, as a four-core CPU's budget needs; the settings are `x264_settings` in
`engine/src/stream.rs`. ffmpeg (`stream/encode.sh`) copies that video untouched and
only encodes the audio. Replayed pictures are the same stream: the same SPS and PPS, the
same GOP, CBR with NAL HRD throughout.

## Channel side (what the channel's owner sets up)

- Live streaming must be enabled on the channel: channel verification, no live
  restrictions in the last 90 days, age 16+. First activation can take up to 24 h.
- Create a stream key in YouTube Studio, Go live, Stream. Use a **persistent key**
  for a 24/7 stream and turn on "auto-start" so reconnects resume the same broadcast.
- 24/7 streams are allowed; there is no published length limit. **Streams longer
  than 12 hours are not archived as a VOD**, so the 24/7 broadcast leaves no replay.
  That is acceptable for a radio-style stream.
- Music must be your own or licensed for streaming, or the stream can be matched and taken down.
- Mark the broadcast "not made for kids" unless it is, since that disables chat.

## Sources

- https://support.google.com/youtube/answer/2853702 (encoder settings, bitrate table)
- https://developers.google.com/youtube/v3/live/guides/ingestion-protocol-comparison
- https://developers.google.com/youtube/v3/live/guides/rtmps-ingestion
- https://support.google.com/youtube/answer/10364924 (RTMPS)
- https://support.google.com/youtube/answer/2474026 (enabling live)
- https://support.google.com/youtube/answer/6247592 (12 hour archive limit)
- https://callaba.io/set-up-srt-ingest-to-youtube (SRT is enabled per account by support)

## Latency

YouTube sets the delay per stream, under Stream settings, Stream latency: normal, low or
ultra-low. Ultra-low keeps most viewers under 5 seconds but makes buffering more likely,
and live automatic captions need normal latency
([YouTube Help, checked 2026-09-25](https://support.google.com/youtube/answer/7444635)).
The stream's own share, from a block placed to the frame reaching the encoder, is about
0.4 s (docs/architecture.md, "The feed"), plus 0.75 s the engine's encoder holds each picture.
