package net.experimentiale.renderfox.observer;

/** An engine's host and port, as typed: {@code host}, {@code host:port} or {@code [v6]:port}. */
record Address(String host, int port) {
    static Address parse(String text) {
        String s = text.trim();
        String host = s;
        String port = null;
        if (s.startsWith("[")) {
            int close = s.indexOf(']');
            if (close < 0) throw new IllegalArgumentException("'" + text + "' opens a [ it never closes");
            host = s.substring(1, close);
            if (s.length() > close + 1) {
                if (s.charAt(close + 1) != ':') throw new IllegalArgumentException("'" + text + "' is not host:port");
                port = s.substring(close + 2);
            }
        } else if (s.indexOf(':') >= 0 && s.indexOf(':') == s.lastIndexOf(':')) {
            host = s.substring(0, s.indexOf(':'));
            port = s.substring(s.indexOf(':') + 1);
        }
        if (host.isEmpty()) throw new IllegalArgumentException("'" + text + "' has no host");
        if (port == null) return new Address(host, RenderFoxPlugin.DEFAULT_PORT);
        try {
            int p = Integer.parseInt(port);
            if (p < 1 || p > 65535) throw new NumberFormatException();
            return new Address(host, p);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("'" + port + "' is not a port");
        }
    }

    @Override
    public String toString() {
        return host.indexOf(':') >= 0 ? "[" + host + "]:" + port : host + ":" + port;
    }
}
