package mage.game;

import mage.MageException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

public class GameImplTest {

    @Test
    public void unitTestFastFailRetainsOriginatingCause() {
        RuntimeException originating = new RuntimeException("originating violation");

        MageException checked = GameImpl.newUnitTestFailure(originating);
        IllegalStateException unchecked = GameImpl.uncheckedUnitTestFailure(checked);

        assertEquals("Error in unit tests", checked.getMessage());
        assertSame(originating, checked.getCause());
        assertEquals("Error in unit tests", unchecked.getMessage());
        assertSame(originating, unchecked.getCause());
    }
}
