import components


def test_get_resistance_min_position():
    pot = components.DigitalPot(100000, 100)
    assert pot.get_resistance(1) == 0


def test_get_resistance_max_position():
    pot = components.DigitalPot(100000, 100)
    assert pot.get_resistance(100) == 100000


def test_get_resistance_mid_position():
    pot = components.DigitalPot(100000, 100)
    step = 100000 / 99
    assert pot.get_resistance(50) == step * 49


def test_get_resistance_clamps_below_range():
    pot = components.DigitalPot(100000, 100)
    assert pot.get_resistance(0) == pot.get_resistance(1)
    assert pot.get_resistance(-10) == pot.get_resistance(1)


def test_get_resistance_clamps_above_range():
    pot = components.DigitalPot(100000, 100)
    assert pot.get_resistance(101) == pot.get_resistance(100)
    assert pot.get_resistance(1000) == pot.get_resistance(100)
