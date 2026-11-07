# Outputs

Where the stream goes is set with the `renderfox` tool on the render box, and kept in
`relay/uplink.env` (mode 0600, ignored by git). The engine never sees it: the encoder
publishes to the local relay, and the uplinks copy from there to the ingest, so changing
the output restarts only the uplinks and never interrupts the renderer.

```sh
renderfox output      # where it goes now, with the key cut off
```

A stream key lets anyone who has it broadcast on your channel. `renderfox` asks for it
without echoing and never prints it whole. If it leaks, reset it on the platform and run
the command again.

## YouTube

```sh
renderfox youtube               # asks for the key
renderfox youtube --no-backup   # the primary ingest only
```

The key is under YouTube Studio, **Go live**, **Stream**, **Stream key** (a pasted
`rtmp://...` URL is fine too; the key is taken from its end). By default the stream goes
to both of YouTube's ingests at once, `rtmps://a.rtmps.youtube.com` and
`rtmps://b.rtmps.youtube.com/live2?backup=1`: YouTube plays the primary and switches to
the backup without a break if the primary drops. The two uplinks are
`renderfox-uplink` and `renderfox-uplink-backup`.

In Studio, for a stream meant to run 24/7:

- **Auto-start** on, so the broadcast starts whenever the stream arrives and continues
  after a drop. Auto-stop, where Studio offers it, off.
- **Stream latency**: low (under about 10 s) is a good balance; ultra-low makes buffering
  more likely, and live captions need normal.
- **DVR** off if viewers should only ever see live.
- **Made for kids**: no, unless it is; yes turns chat off.
- Share the channel's `/live` address (`youtube.com/@yourchannel/live`) rather than one
  broadcast's link: after a long outage YouTube may start a new broadcast with a new link,
  and `/live` always points at the current one.
- Streams longer than 12 hours are not kept as a video, so a 24/7 stream leaves no
  replay.

[youtube-ingest.md](youtube-ingest.md) covers what YouTube accepts and how the encoder
matches it.

## Twitch

```sh
renderfox twitch
```

The key is under the Twitch dashboard, **Settings**, **Stream**, **Primary Stream key**.
The stream goes over RTMPS to `ingest.global-contribute.live-video.net`, which routes to
the nearest ingest. Twitch takes at most 6000 kbit/s from most channels, so set
`encoder.bitrateKbps` to 6000 in `renderfox.json` and run `renderfox restart`; 1080p24 at
6 Mbit/s still looks good for a mostly still orbit. Twitch has no backup ingest.

## Any RTMP or RTMPS ingest

```sh
renderfox rtmp 'rtmps://ingest.example.com/live/STREAM-KEY'
renderfox rtmp 'rtmps://a.example.com/live/KEY' --backup 'rtmps://b.example.com/live/KEY'
```

The full URL, key included. A backup ingest, if the service has one, is sent the same
stream at the same time. Typing a URL on the command line leaves it in the shell's
history; clear it afterwards (`history -d`) if that matters. Certificates are always
checked; a service with a private CA can be given its certificate as `CA_FILE=` in
`uplink.env`.

The stream is H.264 High in FLV, CBR, 2 s closed GOP, AAC-LC stereo 128 kbit/s at
44.1 kHz, which every RTMP ingest takes.

## Testing without going live

`tools/test-ingest/` is a stand-in for YouTube's ingest: RTMPS with a certificate for
`localhost`, the same URL shape, recording what arrives. `validate.py` checks a
recording against YouTube's spec.

```sh
sudo ./install.sh --test-ingest
sudo systemctl start renderfox-test-ingest
renderfox rtmp rtmps://localhost:1936/live2/rf-dummy-key
echo "CA_FILE=$PWD/tools/test-ingest/dummy.crt" >> relay/uplink.env
renderfox restart
python3 tools/test-ingest/validate.py      # after a minute or two
```

Point it back at the real ingest (`renderfox youtube`) and stop the test ingest when
done.
