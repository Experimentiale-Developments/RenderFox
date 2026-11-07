package net.experimentiale.renderfox.observer;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.GeneralSecurityException;
import java.util.Base64;
import java.util.logging.Logger;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public final class IdentityTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private final Logger log = Logger.getLogger("IdentityTest");

    @Test
    public void makesAKeyOnceAndKeepsIt() throws Exception {
        Path dir = folder.getRoot().toPath().resolve("RenderFox");
        Identity identity = Identity.loadOrCreate(dir, log);
        assertEquals(32, identity.publicKey().length);
        assertEquals(identity.publicKeyText(), Files.readString(dir.resolve(Identity.PUBLIC_FILE)).trim());
        assertEquals("rw-------", PosixFilePermissions.toString(Files.getPosixFilePermissions(dir.resolve(Identity.PRIVATE_FILE))));
        assertArrayEquals(identity.publicKey(), Identity.loadOrCreate(dir, log).publicKey());
    }

    @Test
    public void signsWithEd25519() throws Exception {
        Identity identity = Identity.loadOrCreate(folder.getRoot().toPath(), log);
        assertEquals(64, identity.sign("hello renderfox".getBytes()).length);
    }

    @Test(expected = GeneralSecurityException.class)
    public void refusesAPublicKeyThatIsNotItsOwn() throws Exception {
        Path dir = folder.getRoot().toPath();
        Identity.loadOrCreate(dir, log);
        Files.writeString(dir.resolve(Identity.PUBLIC_FILE), Base64.getEncoder().encodeToString(new byte[32]));
        Identity.loadOrCreate(dir, log);
    }

    @Test
    public void readsAPairMadeWithOpenssl() throws Exception {
        // A throwaway pair made by docs/pairing.md's openssl recipe: PKCS#8 DER, and the raw public key in base64.
        Path dir = folder.getRoot().toPath();
        byte[] pkcs8 = Base64.getDecoder().decode("MC4CAQAwBQYDK2VwBCIEIFFOqgF8/J/jieuF7RvK6viv8ycoLWlTy7dJP8uSbS4p");
        Files.write(dir.resolve(Identity.PRIVATE_FILE), pkcs8);
        Files.writeString(dir.resolve(Identity.PUBLIC_FILE), "IEU+C9tSw64xUP9XTTsHhNwdUJotqUj/GX0XBzZcoT4=\n");
        assertEquals("IEU+C9tSw64xUP9XTTsHhNwdUJotqUj/GX0XBzZcoT4=", Identity.loadOrCreate(dir, log).publicKeyText());
    }
}
