package net.experimentiale.renderfox.observer;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class AddressTest {
    @Test
    public void takesTheFormsPeopleType() {
        assertEquals(new Address("render.lan", 25590), Address.parse("render.lan"));
        assertEquals(new Address("10.0.0.5", 4000), Address.parse(" 10.0.0.5:4000 "));
        assertEquals(new Address("fd00::5", 25590), Address.parse("[fd00::5]"));
        assertEquals(new Address("fd00::5", 4000), Address.parse("[fd00::5]:4000"));
        assertEquals("[fd00::5]:4000", Address.parse("[fd00::5]:4000").toString());
    }

    @Test(expected = IllegalArgumentException.class)
    public void refusesABadPort() {
        Address.parse("host:99999");
    }

    @Test(expected = IllegalArgumentException.class)
    public void refusesNoHost() {
        Address.parse(":25590");
    }
}
