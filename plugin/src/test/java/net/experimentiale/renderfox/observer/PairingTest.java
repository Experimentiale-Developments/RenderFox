package net.experimentiale.renderfox.observer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.security.KeyFactory;
import java.security.Signature;
import java.security.spec.X509EncodedKeySpec;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.logging.Logger;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public final class PairingTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    /** The same vector is in the engine's net.rs tests: both sides must agree on it. */
    @Test
    public void proofIsSharedWithTheEngine() throws Exception {
        byte[] nonce = new byte[32];
        for (int i = 0; i < 32; i++) nonce[i] = (byte) i;
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 7);
        assertEquals("1136a1e22f523b7a87a8dd6a13808c75ef26a96e2f27c919a86f3e3a17c188d2",
                HexFormat.of().formatHex(Pairing.proof("abcde-fghjk", nonce, key)));
    }

    @Test
    public void codesSurviveHowPeopleTypeThem() {
        assertEquals("7KQ2MX9PRT", Pairing.normalize(" 7kq2m-x9prt "));
        assertEquals("01110", Pairing.normalize("O1L-Io"));
    }

    @Test
    public void requestCarriesTheKeyTheProofAndASignatureOverBoth() throws Exception {
        Identity identity = Identity.loadOrCreate(folder.getRoot().toPath(), Logger.getLogger("PairingTest"));
        byte[] nonce = new byte[32];
        Arrays.fill(nonce, (byte) 3);
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(Pairing.request(identity, "AAAAA-BBBBB", nonce)));
        Wire.Frame frame = Wire.read(in, 4096);
        assertEquals(Wire.PAIR, frame.type());
        assertEquals(4 + 32 + 32 + 64, frame.payload().length);
        byte[] key = Arrays.copyOfRange(frame.payload(), 4, 36);
        byte[] proof = Arrays.copyOfRange(frame.payload(), 36, 68);
        byte[] signature = Arrays.copyOfRange(frame.payload(), 68, 132);
        assertEquals(identity.publicKeyText(), java.util.Base64.getEncoder().encodeToString(key));
        assertEquals(HexFormat.of().formatHex(Pairing.proof("AAAAABBBBB", nonce, key)), HexFormat.of().formatHex(proof));

        byte[] x509 = new byte[44];
        System.arraycopy(HexFormat.of().parseHex("302a300506032b6570032100"), 0, x509, 0, 12);
        System.arraycopy(key, 0, x509, 12, 32);
        Signature verify = Signature.getInstance("Ed25519");
        verify.initVerify(KeyFactory.getInstance("Ed25519").generatePublic(new X509EncodedKeySpec(x509)));
        verify.update(Wire.PAIR_CONTEXT);
        verify.update(nonce);
        verify.update(key);
        verify.update(proof);
        assertTrue(verify.verify(signature));
    }
}
