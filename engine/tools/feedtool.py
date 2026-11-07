"""Record the observer feed once (as the engine would receive it) and replay it into an
engine, so two engines see byte-identical worlds. Entity frames are dropped.

    feedtool.py key test.key             make a key to replay with; prints its public half
    feedtool.py record PORT feed.bin     stand in for the engine while the observer connects
    feedtool.py replay PORT feed.bin test.key
    feedtool.py pair PORT test.key CODE  pair an unpaired engine with the key, as the plugin does

The engine only takes a feed signed by the key it is paired with (docs/protocol.md), so a
replay signs the recorded hello afresh with the given key: pair the test engine with it
first, or put its public half in the test engine's observerKey. The key is Ed25519 in
PKCS#8 DER, the plugin's identity.key format."""
import hashlib, hmac, os, socket, struct, sys, time
from base64 import b64encode
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HELLO, SECTIONS, SNAPSHOT_DONE, ENTITIES, ENTITY_INFO, PING, CHALLENGE, ACCEPT = 1, 2, 3, 4, 5, 6, 7, 8
PAIR, PAIRED, REFUSED = 9, 10, 11
PROTOCOL = 2
CONTEXT = b"RenderFox observer hello v2"
PAIR_CONTEXT = b"RenderFox pair v1"

def frames(sock):
    buf = b""
    while True:
        while len(buf) < 4:
            chunk = sock.recv(1 << 20)
            if not chunk:
                return
            buf += chunk
        (n,) = struct.unpack(">I", buf[:4])
        while len(buf) < 4 + n:
            chunk = sock.recv(1 << 20)
            if not chunk:
                return
            buf += chunk
        yield buf[:4 + n]
        buf = buf[4 + n:]

def frame(kind, payload):
    return struct.pack(">IB", len(payload) + 1, kind) + payload

def public_half(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

def keygen(path):
    key = Ed25519PrivateKey.generate()
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(der)
    print(b64encode(public_half(key)).decode())

def record(port, out):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    # Any nonce will do: the recorded hello is signed again when replayed.
    conn.sendall(frame(CHALLENGE, struct.pack(">i", PROTOCOL) + os.urandom(32)))
    kept = 0
    with open(out, "wb") as f:
        for fr in frames(conn):
            kind = fr[4]
            if kind == HELLO:
                conn.sendall(frame(ACCEPT, b""))
            if kind in (HELLO, SECTIONS, SNAPSHOT_DONE):
                f.write(fr)
                kept += len(fr)
            if kind == SNAPSHOT_DONE:
                break
    conn.close()
    print(f"recorded {kept / 1048576:.1f} MiB")

def replay(port, path, keyfile):
    key = serialization.load_der_private_key(open(keyfile, "rb").read(), password=None)
    data = open(path, "rb").read()
    (n,) = struct.unpack(">I", data[:4])
    # The recorded hello: protocol, the recording server's key, the area; then its signature.
    body = data[5:4 + n - 64]
    body = body[:4] + public_half(key) + body[36:]
    for _ in range(100):
        try:
            s = socket.create_connection(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.5)
    incoming = frames(s)
    challenge = next(incoming)
    if challenge[4] != CHALLENGE:
        sys.exit("the engine did not send a challenge")
    nonce = challenge[9:41]
    s.sendall(frame(HELLO, body + key.sign(CONTEXT + nonce + body)))
    if next(incoming, b"\0\0\0\0\0")[4] != ACCEPT:
        sys.exit("the engine refused the key: its observerKey must be " + b64encode(public_half(key)).decode())
    s.sendall(data[4 + n:])
    ping = frame(PING, b"")
    try:
        while True:
            time.sleep(5)
            s.sendall(ping)
    except OSError:
        pass

def normalize(code):
    return "".join({"O": "0", "I": "1", "L": "1"}.get(c, c) for c in code.upper() if c.isascii() and c.isalnum())

def pair(port, keyfile, code):
    key = serialization.load_der_private_key(open(keyfile, "rb").read(), password=None)
    public = public_half(key)
    s = socket.create_connection(("127.0.0.1", port))
    incoming = frames(s)
    challenge = next(incoming)
    if challenge[4] != CHALLENGE:
        sys.exit("the engine did not send a challenge")
    nonce = challenge[9:41]
    proof = hmac.new(normalize(code).encode(), PAIR_CONTEXT + nonce + public, hashlib.sha256).digest()
    signature = key.sign(PAIR_CONTEXT + nonce + public + proof)
    s.sendall(frame(PAIR, struct.pack(">i", PROTOCOL) + public + proof + signature))
    answer = next(incoming, None)
    if answer is None:
        sys.exit("the engine closed the connection without answering")
    if answer[4] == PAIRED:
        print("paired")
    elif answer[4] == REFUSED:
        (n,) = struct.unpack(">H", answer[5:7])
        sys.exit("refused: " + answer[7:7 + n].decode())
    else:
        sys.exit(f"unexpected answer type {answer[4]}")

if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    if command == "key":
        keygen(args[0])
    elif command == "record":
        record(int(args[0]), args[1])
    elif command == "pair":
        pair(int(args[0]), args[1], args[2])
    else:
        replay(int(args[0]), args[1], args[2])
