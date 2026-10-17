"""Birthday window check (deployed with Regression Lab through 2026-10-25)."""

from __future__ import annotations

from datetime import datetime

from src.birthday_ines import BIRTHDAY_LAST_DAY, PARIS, is_birthday_window_open


def test_birthday_window_open_through_oct_25():
    assert is_birthday_window_open(datetime(2026, 10, 25, 23, 59, tzinfo=PARIS))
    assert BIRTHDAY_LAST_DAY.isoformat() == "2026-10-25"


def test_birthday_window_closed_from_oct_26():
    assert not is_birthday_window_open(datetime(2026, 10, 26, 0, 0, tzinfo=PARIS))
