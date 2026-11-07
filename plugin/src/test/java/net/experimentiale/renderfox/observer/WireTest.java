package net.experimentiale.renderfox.observer;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import org.junit.Test;

public final class WireTest {

    @Test
    public void protocolVersionMatches() {
        assertEquals(2, Wire.PROTOCOL);
        assertEquals(32, Wire.NONCE_BYTES);
        assertEquals("RenderFox observer hello v2", new String(Wire.HELLO_CONTEXT, StandardCharsets.US_ASCII));
        assertEquals("RenderFox pair v1", new String(Wire.PAIR_CONTEXT, StandardCharsets.US_ASCII));
        assertEquals(9, Wire.PAIR);
        assertEquals(10, Wire.PAIRED);
        assertEquals(11, Wire.REFUSED);
    }

    @Test
    public void readsAnyFrame() throws IOException {
        byte[] framed = Wire.message(Wire.REFUSED, out -> out.writeUTF("wrong code"));
        Wire.Frame frame = Wire.read(new DataInputStream(new ByteArrayInputStream(framed)), 4096);
        assertEquals(Wire.REFUSED, frame.type());
        assertEquals("wrong code", new DataInputStream(new ByteArrayInputStream(frame.payload())).readUTF());
    }

    @Test(expected = IOException.class)
    public void refusesAnOversizedFrame() throws IOException {
        Wire.read(new DataInputStream(new ByteArrayInputStream(Wire.frame(Wire.PING, new byte[100]))), 10);
    }

    @Test
    public void framesMessageCorrectly() {
        byte[] payload = "test-payload".getBytes(StandardCharsets.UTF_8);
        byte[] framed = Wire.frame(Wire.HELLO, payload);

        ByteBuffer buf = ByteBuffer.wrap(framed);
        int length = buf.getInt();
        assertEquals(payload.length + 1, length);
        byte type = buf.get();
        assertEquals(Wire.HELLO, type);
        byte[] body = new byte[payload.length];
        buf.get(body);
        assertArrayEquals(payload, body);
    }

    @Test
    public void rawBodyWritesExpectedBytes() {
        byte[] result = Wire.raw(out -> {
            out.writeInt(42);
            out.writeUTF("hello");
        });

        ByteBuffer buf = ByteBuffer.wrap(result);
        assertEquals(42, buf.getInt());
        short utfLen = buf.getShort();
        assertEquals(5, utfLen);
        byte[] utfBytes = new byte[utfLen];
        buf.get(utfBytes);
        assertEquals("hello", new String(utfBytes, StandardCharsets.UTF_8));
    }

    @Test
    public void deflatedCompressesAndDecompresses() throws Exception {
        byte[] original = "test section data repeated repeated repeated 123456789".getBytes(StandardCharsets.UTF_8);
        byte[] framed = Wire.deflated(Wire.SECTIONS, out -> out.write(original));

        ByteBuffer buf = ByteBuffer.wrap(framed);
        int len = buf.getInt();
        byte type = buf.get();
        assertEquals(Wire.SECTIONS, type);

        byte[] compressed = new byte[len - 1];
        buf.get(compressed);

        InflaterInputStream inflater = new InflaterInputStream(new ByteArrayInputStream(compressed), new Inflater());
        ByteArrayOutputStream decompressed = new ByteArrayOutputStream();
        inflater.transferTo(decompressed);

        assertArrayEquals(original, decompressed.toByteArray());
    }

    @Test
    public void expectReturnsPayloadOnSuccess() throws IOException {
        byte[] payload = new byte[]{10, 20, 30};
        byte[] framed = Wire.frame(Wire.CHALLENGE, payload);

        DataInputStream in = new DataInputStream(new ByteArrayInputStream(framed));
        byte[] got = Wire.expect(in, Wire.CHALLENGE, 3);
        assertArrayEquals(payload, got);
    }

    @Test
    public void expectFailsOnTypeMismatch() {
        byte[] payload = new byte[]{1, 2, 3};
        byte[] framed = Wire.frame(Wire.PING, payload);

        DataInputStream in = new DataInputStream(new ByteArrayInputStream(framed));
        try {
            Wire.expect(in, Wire.ACCEPT, 3);
            fail("Expected IOException on type mismatch");
        } catch (IOException e) {
            // expected
        }
    }

    @Test
    public void expectFailsOnLengthMismatch() {
        byte[] payload = new byte[]{1, 2, 3};
        byte[] framed = Wire.frame(Wire.CHALLENGE, payload);

        DataInputStream in = new DataInputStream(new ByteArrayInputStream(framed));
        try {
            Wire.expect(in, Wire.CHALLENGE, 5);
            fail("Expected IOException on length mismatch");
        } catch (IOException e) {
            // expected
        }
    }
}
