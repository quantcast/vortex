// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Set;
import org.junit.jupiter.api.Test;

public final class SessionTest {

    @Test
    public void knownEditionsIncludeTheCoreEditionsAndTheListLayoutEdition() {
        Set<String> known = Session.create().knownEditions();

        assertTrue(known.contains("core2025.10.0"), known::toString);
        assertTrue(known.contains("core2026.08.3"), known::toString);
        assertTrue(known.contains("qc2026.11.0"), known::toString);
        assertFalse(known.contains("core2099.01.0"), known::toString);
    }

    @Test
    public void everyKnownEditionCanBeEnabled() {
        Session session = Session.create();

        for (String edition : session.knownEditions()) {
            assertDoesNotThrow(() -> session.enableEdition(edition), edition);
        }
    }

    @Test
    public void anEditionThatIsntKnownCantBeEnabled() {
        Session session = Session.create();

        assertFalse(session.knownEditions().contains("core2099.01.0"));
        assertThrows(RuntimeException.class, () -> session.enableEdition("core2099.01.0"));
    }
}
