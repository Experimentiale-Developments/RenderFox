package net.experimentiale.renderfox.observer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.DataInputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/** The link against a stand-in engine that answers the first connection's PAIR. */
public final class LinkTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private final Logger log = Logger.getLogger("LinkTest");

    private static byte[] challenge(byte fill) {
        return Wire.message(Wire.CHALLENGE, out -> {
            out.writeInt(Wire.PROTOCOL);
            for (int i = 0; i < Wire.NONCE_BYTES; i++) out.writeByte(fill);
        });
    }

    @Test
    public void pairsThenSaysHello() throws Exception {
        Identity identity = Identity.loadOrCreate(folder.getRoot().toPath(), log);
        List<Byte> seen = Collections.synchronizedList(new ArrayList<>());
        CompletableFuture<Void> paired = new CompletableFuture<>();
        CompletableFuture<Void> connected = new CompletableFuture<>();
        CompletableFuture<Void> checked = new CompletableFuture<>();
        try (ServerSocket engine = new ServerSocket(0)) {
            Thread fake = Thread.ofPlatform().start(() -> {
                try {
                    try (Socket s = engine.accept()) {
                        OutputStream out = s.getOutputStream();
                        out.write(challenge((byte) 1));
                        Wire.Frame pair = Wire.read(new DataInputStream(s.getInputStream()), 4096);
                        seen.add(pair.type());
                        out.write(Wire.frame(Wire.PAIRED, new byte[0]));
                    }
                    try (Socket s = engine.accept()) {
                        OutputStream out = s.getOutputStream();
                        out.write(challenge((byte) 2));
                        Wire.Frame hello = Wire.read(new DataInputStream(s.getInputStream()), 4096);
                        seen.add(hello.type());
                        out.write(Wire.frame(Wire.ACCEPT, new byte[0]));
                        checked.get(5, TimeUnit.SECONDS);
                    }
                } catch (Exception e) {
                    connected.completeExceptionally(e);
                }
            });
            Link link = new Link("127.0.0.1", engine.getLocalPort(), nonce -> Wire.frame(Wire.HELLO, nonce), identity,
                    () -> connected.complete(null), log);
            link.pair("AAAAA-BBBBB", () -> paired.complete(null), reason -> paired.completeExceptionally(new AssertionError(reason)));
            link.start();
            paired.get(10, TimeUnit.SECONDS);
            connected.get(10, TimeUnit.SECONDS);
            assertTrue(link.connected());
            checked.complete(null);
            link.close();
            fake.join(5000);
        }
        assertEquals(List.of(Wire.PAIR, Wire.HELLO), seen);
    }

    @Test
    public void reportsTheEnginesRefusal() throws Exception {
        Identity identity = Identity.loadOrCreate(folder.getRoot().toPath(), log);
        CompletableFuture<String> failed = new CompletableFuture<>();
        try (ServerSocket engine = new ServerSocket(0)) {
            Thread.ofPlatform().start(() -> {
                try (Socket s = engine.accept()) {
                    s.getOutputStream().write(challenge((byte) 1));
                    Wire.read(new DataInputStream(s.getInputStream()), 4096);
                    s.getOutputStream().write(Wire.message(Wire.REFUSED, out -> out.writeUTF("tried to pair with the wrong code")));
                } catch (Exception e) {
                    failed.completeExceptionally(e);
                }
            });
            Link link = new Link("127.0.0.1", engine.getLocalPort(), nonce -> Wire.frame(Wire.HELLO, nonce), identity, () -> {}, log);
            link.pair("WRONG-CODE0", () -> failed.complete("paired"), failed::complete);
            link.start();
            String reason = failed.get(10, TimeUnit.SECONDS);
            link.close();
            assertTrue(reason, reason.endsWith("refused to pair: tried to pair with the wrong code"));
        }
    }

    @Test
    public void reportsAnEngineThatCannotBeReached() throws Exception {
        Identity identity = Identity.loadOrCreate(folder.getRoot().toPath(), log);
        int port;
        try (ServerSocket closed = new ServerSocket(0)) {
            port = closed.getLocalPort();
        }
        CompletableFuture<String> failed = new CompletableFuture<>();
        Link link = new Link("127.0.0.1", port, nonce -> Wire.frame(Wire.HELLO, nonce), identity, () -> {}, log);
        link.pair("AAAAA-BBBBB", () -> failed.complete("paired"), failed::complete);
        link.start();
        String reason = failed.get(10, TimeUnit.SECONDS);
        link.close();
        assertTrue(reason, reason.startsWith("Render Fox engine unreachable at 127.0.0.1:" + port));
    }
}
