package net.experimentiale.renderfox.observer;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.security.GeneralSecurityException;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * Outbound connection to the engine. The plugin dials out rather than listening so
 * that nothing new has to be opened on the game server's side, which is often a
 * container or a host that only forwards the game port.
 *
 * Every connection starts with a handshake: the engine sends a fresh nonce, the plugin
 * answers with its hello signed by this server's key, and the engine accepts it only if
 * that key is the one it is paired with. To pair, the plugin answers one challenge with
 * a PAIR message instead (Pairing), then reconnects and says hello as usual.
 *
 * Losing a terrain message would leave the engine's copy of the world silently wrong,
 * so terrain is never dropped: if the backlog passes the hard limit the connection is
 * cut instead, and the reconnect resends everything. Entities travel in a lane of their
 * own that is always sent first, so a player never waits behind a burst of chunks;
 * their position frames are superseded twenty times a second, so those are shed if that
 * lane ever backs up.
 */
final class Link implements Runnable {
    private static final long FAST_LIMIT = 1L << 20;
    private static final long HARD_LIMIT = 96L << 20;
    private static final int HANDSHAKE_TIMEOUT_MS = 10_000;

    interface Hello {
        /** The framed HELLO message, signed over the engine's nonce. */
        byte[] signed(byte[] nonce) throws GeneralSecurityException;
    }

    private static final class Refused extends IOException {
        Refused(String message) {
            super(message);
        }
    }

    /** Thrown once paired: reconnect at once, this time with a hello. */
    private static final class Paired extends IOException {
    }

    /** A pairing to try on the next connection, and whom to tell how it went. */
    private record PairRequest(String code, Runnable paired, Consumer<String> failed) {
    }

    private final InetSocketAddress address;
    private final Hello hello;
    private final Identity identity;
    private final String key;
    private final Runnable onConnect;
    private final Logger log;
    private final Object lock = new Object();
    // Guarded by lock.
    private final ArrayDeque<byte[]> fast = new ArrayDeque<>();
    private final ArrayDeque<byte[]> terrain = new ArrayDeque<>();
    private long fastBytes;
    private final AtomicLong backlog = new AtomicLong();
    private final Thread thread;

    private volatile boolean connected;
    private volatile boolean closed;
    private volatile Socket socket;
    private final AtomicLong sent = new AtomicLong();
    private volatile long lastProgress = System.nanoTime();
    private volatile PairRequest pairing;

    Link(String host, int port, Hello hello, Identity identity, Runnable onConnect, Logger log) {
        this.address = InetSocketAddress.createUnresolved(host, port);
        this.hello = hello;
        this.identity = identity;
        this.key = identity.publicKeyText();
        this.onConnect = onConnect;
        this.log = log;
        this.thread = Thread.ofPlatform().name("RenderFox-link").daemon().unstarted(this);
    }

    void start() {
        thread.start();
    }

    boolean connected() {
        return connected;
    }

    boolean pairing() {
        return pairing != null;
    }

    String address() {
        return address.getHostString() + ":" + address.getPort();
    }

    /**
     * Pairs on the next connection, cutting the current one for it. The callbacks run on
     * the link's thread; exactly one of them runs.
     */
    void pair(String code, Runnable paired, Consumer<String> failed) {
        pairing = new PairRequest(code, paired, failed);
        drop();
    }

    long backlog() {
        return backlog.get();
    }

    long sentBytes() {
        return sent.get();
    }

    void sendTerrain(byte[] message) {
        reliable(terrain, message);
    }

    /** Entity info is sent once per change, so unlike a position it is never dropped. */
    void sendInfo(byte[] message) {
        reliable(fast, message);
    }

    void sendVolatile(byte[] message) {
        if (!connected) return;
        synchronized (lock) {
            if (fastBytes > FAST_LIMIT) return;
            fast.add(message);
            fastBytes += message.length;
            backlog.addAndGet(message.length);
            lock.notifyAll();
        }
    }

    private void reliable(ArrayDeque<byte[]> lane, byte[] message) {
        if (!connected) return;
        if (backlog.addAndGet(message.length) > HARD_LIMIT) {
            log.warning("Engine is not keeping up with terrain, reconnecting to resync");
            drop();
            return;
        }
        synchronized (lock) {
            lane.add(message);
            if (lane == fast) fastBytes += message.length;
            lock.notifyAll();
        }
    }

    void close() {
        closed = true;
        drop();
        thread.interrupt();
    }

    private void drop() {
        connected = false;
        Socket s = socket;
        if (s != null) {
            try {
                s.close();
            } catch (IOException ignored) {
                // closing to force the writer loop out; nothing to recover
            }
        }
        synchronized (lock) {
            lock.notifyAll();
        }
    }

    @Override
    public void run() {
        long backoff = 1000;
        String problem = null;
        while (!closed) {
            String failure = null;
            PairRequest attempt = pairing;
            try (Socket s = new Socket()) {
                socket = s;
                s.connect(new InetSocketAddress(address.getHostString(), address.getPort()), 5000);
                s.setTcpNoDelay(true);
                s.setKeepAlive(true);
                OutputStream out = new BufferedOutputStream(s.getOutputStream(), 1 << 16);
                handshake(s, out, attempt);
                synchronized (lock) {
                    fast.clear();
                    terrain.clear();
                    fastBytes = 0;
                }
                backlog.set(0);
                lastProgress = System.nanoTime();
                watchForClose(s);
                connected = true;
                log.info("Connected to Render Fox engine at " + address.getHostString() + ":" + address.getPort()
                        + ", which accepted this server's key");
                problem = null;
                backoff = 1000;
                onConnect.run();
                pump(out);
            } catch (Paired e) {
                problem = null;
                backoff = 1000;
                continue;
            } catch (Refused e) {
                failure = e.getMessage();
            } catch (IOException e) {
                // pair() cuts the connection to make room for itself: that is no failure.
                if (!closed && !(attempt == null && pairing != null)) {
                    failure = "Render Fox engine unreachable at " + address() + " (" + e.getMessage() + ")";
                }
            } catch (GeneralSecurityException e) {
                failure = "Cannot sign the handshake with the Render Fox engine: " + e;
            } catch (InterruptedException e) {
                return;
            } finally {
                connected = false;
                socket = null;
            }
            if (failure != null) {
                if (attempt != null && pairing == attempt) {
                    pairing = null;
                    attempt.failed().accept(failure);
                }
                problem = report(problem, failure);
            }
            try {
                pause(backoff);
            } catch (InterruptedException e) {
                return;
            }
            backoff = Math.min(backoff * 2, 10_000);
        }
    }

    /** Sleeps between attempts, waking early for a pairing or for close. */
    private void pause(long millis) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(millis);
        synchronized (lock) {
            long left;
            while (!closed && pairing == null && (left = deadline - System.nanoTime()) > 0) {
                TimeUnit.NANOSECONDS.timedWait(lock, left);
            }
        }
    }

    /** Logs a problem once, not on every retry. */
    private String report(String last, String problem) {
        if (!problem.equals(last)) log.warning(problem);
        return problem;
    }

    private void handshake(Socket s, OutputStream out, PairRequest request) throws IOException, GeneralSecurityException {
        s.setSoTimeout(HANDSHAKE_TIMEOUT_MS);
        DataInputStream in = new DataInputStream(s.getInputStream());
        byte[] challenge;
        try {
            challenge = Wire.expect(in, Wire.CHALLENGE, 4 + Wire.NONCE_BYTES);
        } catch (SocketTimeoutException e) {
            throw new Refused("The Render Fox engine took the connection but sent no challenge: it feeds one game server"
                    + " at a time and another is connected, or it is older than this plugin");
        }
        int protocol = ByteBuffer.wrap(challenge).getInt();
        if (protocol != Wire.PROTOCOL) {
            throw new Refused("The Render Fox engine speaks protocol " + protocol + " and this plugin " + Wire.PROTOCOL
                    + "; update whichever is older");
        }
        byte[] nonce = Arrays.copyOfRange(challenge, 4, challenge.length);
        if (request != null) {
            pair(in, out, nonce, request);
        }
        out.write(hello.signed(nonce));
        out.flush();
        try {
            Wire.expect(in, Wire.ACCEPT, 0);
        } catch (EOFException e) {
            throw new Refused("The Render Fox engine at " + address() + " refused this server: it is paired with another"
                    + " server, or with none yet. Pair it with /renderfox pair " + address() + " <code>, taking the code"
                    + " from `renderfox pair` on the render box. This server's key is " + key);
        }
        s.setSoTimeout(0);
    }

    /**
     * Always throws: Paired to reconnect with a hello, or Refused with the engine's reason,
     * which the loop reports to whoever asked.
     */
    private void pair(DataInputStream in, OutputStream out, byte[] nonce, PairRequest request)
            throws IOException, GeneralSecurityException {
        out.write(Pairing.request(identity, request.code(), nonce));
        out.flush();
        Wire.Frame answer;
        try {
            answer = Wire.read(in, 4096);
        } catch (EOFException e) {
            throw new Refused("The Render Fox engine closed the connection instead of answering the pairing request: it is"
                    + " older than this plugin");
        }
        if (answer.type() == Wire.PAIRED) {
            pairing = null;
            log.info("Paired with the Render Fox engine at " + address() + "; it now streams this server");
            request.paired().run();
            throw new Paired();
        }
        String reason = answer.type() == Wire.REFUSED
                ? new DataInputStream(new ByteArrayInputStream(answer.payload())).readUTF()
                : "it answered with message type " + answer.type();
        throw new Refused("The Render Fox engine at " + address() + " refused to pair: " + reason);
    }

    /**
     * After the handshake the engine sends nothing, so a read only returns when it closes
     * its end. A writer can otherwise sit blocked on a dead peer's full window indefinitely.
     */
    private void watchForClose(Socket s) {
        Thread.ofPlatform().name("RenderFox-link-reader").daemon().start(() -> {
            try {
                while (s.getInputStream().read() >= 0) {
                    // nothing to read by protocol; keep draining until close
                }
            } catch (IOException ignored) {
                // the socket closing is exactly what we are waiting for
            }
            if (socket == s) drop();
        });
    }

    /** Main thread, periodically: cut a connection whose writes stopped progressing. */
    void watchdog() {
        if (connected && backlog.get() > 0 && System.nanoTime() - lastProgress > 20_000_000_000L) {
            log.warning("Render Fox engine stopped reading for 20 s, reconnecting");
            drop();
        }
    }

    private void pump(OutputStream out) throws IOException, InterruptedException {
        byte[] ping = Wire.message(Wire.PING, o -> {});
        while (connected) {
            byte[] message;
            boolean urgent;
            boolean more;
            synchronized (lock) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                long left;
                while (connected && fast.isEmpty() && terrain.isEmpty() && (left = deadline - System.nanoTime()) > 0) {
                    TimeUnit.NANOSECONDS.timedWait(lock, left);
                }
                if (!connected) return;
                urgent = !fast.isEmpty();
                message = urgent ? fast.poll() : terrain.poll();
                if (urgent) fastBytes -= message.length;
                more = !fast.isEmpty() || !terrain.isEmpty();
            }
            if (message == null) {
                out.write(ping);
                out.flush();
                lastProgress = System.nanoTime();
                continue;
            }
            out.write(message);
            backlog.addAndGet(-message.length);
            sent.addAndGet(message.length);
            if (urgent || !more) out.flush();
            lastProgress = System.nanoTime();
        }
    }
}
