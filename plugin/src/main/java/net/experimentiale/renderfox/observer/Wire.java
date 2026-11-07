package net.experimentiale.renderfox.observer;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/**
 * Framing shared with the engine (docs/protocol.md). Every message is
 * {@code int length, byte type, payload}; length counts the type byte. Payloads of
 * {@link #SECTIONS} are deflated because a chunk of terrain is mostly repeats.
 */
final class Wire {
    static final int PROTOCOL = 2;
    static final int NONCE_BYTES = 32;
    /** Signed ahead of the engine's nonce and the hello, so the signature means nothing elsewhere. */
    static final byte[] HELLO_CONTEXT = "RenderFox observer hello v2".getBytes(StandardCharsets.US_ASCII);
    static final byte[] PAIR_CONTEXT = "RenderFox pair v1".getBytes(StandardCharsets.US_ASCII);

    static final byte HELLO = 1;
    static final byte SECTIONS = 2;
    static final byte SNAPSHOT_DONE = 3;
    static final byte ENTITIES = 4;
    static final byte ENTITY_INFO = 5;
    static final byte PING = 6;
    /** Engine to plugin: the nonce the hello must sign. */
    static final byte CHALLENGE = 7;
    /** Engine to plugin: the hello was accepted. */
    static final byte ACCEPT = 8;
    /** Instead of a hello, on an unpaired engine: this server's key and proof of the code. */
    static final byte PAIR = 9;
    /** Engine to plugin: paired; it closes, and the plugin reconnects with a hello. */
    static final byte PAIRED = 10;
    /** Engine to plugin: the pairing was refused, with the reason. */
    static final byte REFUSED = 11;

    private Wire() {
    }

    interface Body {
        void write(DataOutputStream out) throws IOException;
    }

    static byte[] message(byte type, Body body) {
        return frame(type, raw(body));
    }

    static byte[] deflated(byte type, Body body) {
        byte[] plain = raw(body);
        ByteArrayOutputStream packed = new ByteArrayOutputStream(plain.length / 4 + 64);
        Deflater deflater = new Deflater(Deflater.BEST_SPEED);
        try (DeflaterOutputStream z = new DeflaterOutputStream(packed, deflater)) {
            z.write(plain);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            deflater.end();
        }
        return frame(type, packed.toByteArray());
    }

    /** The payload of the engine's next message, which must be of the given type. */
    static byte[] expect(DataInputStream in, byte type, int length) throws IOException {
        int framed = in.readInt();
        if (framed != length + 1) throw new IOException("unexpected handshake from the engine (an older engine?)");
        byte got = in.readByte();
        if (got != type) throw new IOException("expected message type " + type + " from the engine, got " + got);
        byte[] payload = new byte[length];
        in.readFully(payload);
        return payload;
    }

    record Frame(byte type, byte[] payload) {
    }

    /** The engine's next message, whatever its type. */
    static Frame read(DataInputStream in, int maxLength) throws IOException {
        int framed = in.readInt();
        if (framed < 1 || framed > maxLength + 1) throw new IOException("unexpected message from the engine (" + framed + " bytes)");
        byte type = in.readByte();
        byte[] payload = new byte[framed - 1];
        in.readFully(payload);
        return new Frame(type, payload);
    }

    static byte[] raw(Body body) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(256);
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            body.write(out);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return bytes.toByteArray();
    }

    static byte[] frame(byte type, byte[] payload) {
        int length = payload.length + 1;
        byte[] framed = new byte[length + 4];
        framed[0] = (byte) (length >>> 24);
        framed[1] = (byte) (length >>> 16);
        framed[2] = (byte) (length >>> 8);
        framed[3] = (byte) length;
        framed[4] = type;
        System.arraycopy(payload, 0, framed, 5, payload.length);
        return framed;
    }
}
