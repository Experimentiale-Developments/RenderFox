package net.experimentiale.renderfox.observer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import java.util.logging.Logger;

/**
 * This server's Ed25519 key pair: {@code identity.key} (PKCS#8 DER, readable by the server's
 * user only) and {@code identity.pub} (the raw public key in base64) in the plugin's folder.
 * The engine trusts exactly one public key, so only the server holding the matching private
 * key can feed the stream; a copy of the plugin anywhere else is refused. Never copy
 * identity.key to another server.
 */
final class Identity {
    static final String PRIVATE_FILE = "identity.key";
    static final String PUBLIC_FILE = "identity.pub";
    // An Ed25519 SubjectPublicKeyInfo is this fixed prefix followed by the raw 32-byte key.
    private static final byte[] X509_PREFIX = {0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00};

    private final PrivateKey privateKey;
    private final byte[] publicKey;

    private Identity(PrivateKey privateKey, byte[] publicKey) {
        this.privateKey = privateKey;
        this.publicKey = publicKey;
    }

    /**
     * Loads the pair, or makes one on first start. A pair that does not match is an error
     * rather than a reason to make a new one: a new key would silently lock this server out
     * of an engine paired with the old one.
     */
    static Identity loadOrCreate(Path dir, Logger log) throws GeneralSecurityException, IOException {
        Path privateFile = dir.resolve(PRIVATE_FILE);
        Path publicFile = dir.resolve(PUBLIC_FILE);
        if (Files.notExists(privateFile)) {
            KeyPair pair = KeyPairGenerator.getInstance("Ed25519").generateKeyPair();
            byte[] raw = raw(pair.getPublic());
            Files.createDirectories(dir);
            writePrivate(privateFile, pair.getPrivate().getEncoded());
            Files.writeString(publicFile, Base64.getEncoder().encodeToString(raw) + "\n");
            log.info("Made this server's Render Fox key in " + PRIVATE_FILE + " and " + PUBLIC_FILE);
            return new Identity(pair.getPrivate(), raw);
        }
        PrivateKey privateKey = KeyFactory.getInstance("Ed25519")
                .generatePrivate(new PKCS8EncodedKeySpec(Files.readAllBytes(privateFile)));
        if (Files.notExists(publicFile)) throw new GeneralSecurityException(PUBLIC_FILE + " is missing beside " + PRIVATE_FILE);
        byte[] raw = Base64.getDecoder().decode(Files.readString(publicFile, StandardCharsets.US_ASCII).trim());
        if (raw.length != 32) throw new GeneralSecurityException(PUBLIC_FILE + " does not hold a 32-byte key");
        Identity identity = new Identity(privateKey, raw);
        byte[] probe = "render fox key check".getBytes(StandardCharsets.US_ASCII);
        Signature check = Signature.getInstance("Ed25519");
        check.initVerify(identity.verifier());
        check.update(probe);
        if (!check.verify(identity.sign(probe))) {
            throw new GeneralSecurityException(PUBLIC_FILE + " does not belong to " + PRIVATE_FILE);
        }
        return identity;
    }

    byte[] publicKey() {
        return publicKey.clone();
    }

    String publicKeyText() {
        return Base64.getEncoder().encodeToString(publicKey);
    }

    byte[] sign(byte[] message) throws GeneralSecurityException {
        Signature signer = Signature.getInstance("Ed25519");
        signer.initSign(privateKey);
        signer.update(message);
        return signer.sign();
    }

    private PublicKey verifier() throws GeneralSecurityException {
        byte[] encoded = Arrays.copyOf(X509_PREFIX, X509_PREFIX.length + 32);
        System.arraycopy(publicKey, 0, encoded, X509_PREFIX.length, 32);
        return KeyFactory.getInstance("Ed25519").generatePublic(new X509EncodedKeySpec(encoded));
    }

    private static byte[] raw(PublicKey key) throws GeneralSecurityException {
        byte[] encoded = key.getEncoded();
        if (encoded.length != X509_PREFIX.length + 32 || !Arrays.equals(encoded, 0, X509_PREFIX.length, X509_PREFIX, 0, X509_PREFIX.length)) {
            throw new GeneralSecurityException("unexpected Ed25519 public key encoding");
        }
        return Arrays.copyOfRange(encoded, X509_PREFIX.length, encoded.length);
    }

    /** Created owner-only where the file system allows it, so the key is never readable by others, even briefly. */
    private static void writePrivate(Path file, byte[] bytes) throws IOException {
        try {
            Files.createFile(file, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
        } catch (UnsupportedOperationException e) {
            Files.createFile(file);
        } catch (FileAlreadyExistsException e) {
            throw new IOException(file + " appeared while making a new key", e);
        }
        Files.write(file, bytes);
    }
}
