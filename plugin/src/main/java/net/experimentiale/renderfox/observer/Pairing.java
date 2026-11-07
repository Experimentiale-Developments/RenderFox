package net.experimentiale.renderfox.observer;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.Locale;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * The PAIR message: this server's key, an HMAC proving it knows the engine's one-time code,
 * and a signature proving it holds the key. The code itself never crosses the wire.
 */
final class Pairing {
    private Pairing() {
    }

    /** As the engine reads a code: upper case, no separators, O as 0 and I or L as 1. */
    static String normalize(String code) {
        StringBuilder out = new StringBuilder(code.length());
        for (char c : code.toUpperCase(Locale.ROOT).toCharArray()) {
            if (c >= '0' && c <= '9' || c >= 'A' && c <= 'Z') {
                out.append(switch (c) {
                    case 'O' -> '0';
                    case 'I', 'L' -> '1';
                    default -> c;
                });
            }
        }
        return out.toString();
    }

    static byte[] proof(String code, byte[] nonce, byte[] publicKey) throws GeneralSecurityException {
        String normalized = normalize(code);
        if (normalized.isEmpty()) throw new GeneralSecurityException("the pairing code is empty");
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(normalized.getBytes(StandardCharsets.US_ASCII), "HmacSHA256"));
        mac.update(Wire.PAIR_CONTEXT);
        mac.update(nonce);
        mac.update(publicKey);
        return mac.doFinal();
    }

    /** The framed PAIR message for this connection's nonce. */
    static byte[] request(Identity identity, String code, byte[] nonce) throws GeneralSecurityException {
        byte[] key = identity.publicKey();
        byte[] proof = proof(code, nonce, key);
        byte[] signed = Wire.raw(out -> {
            out.write(Wire.PAIR_CONTEXT);
            out.write(nonce);
            out.write(key);
            out.write(proof);
        });
        byte[] signature = identity.sign(signed);
        return Wire.message(Wire.PAIR, out -> {
            out.writeInt(Wire.PROTOCOL);
            out.write(key);
            out.write(proof);
            out.write(signature);
        });
    }
}
