import sys
from pathlib import Path
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import input_control


def test_tap_key_shifted_presses_shift_then_key_then_releases():
    with patch("input_control.keyboard") as mock_kb, \
         patch("input_control.sleep_interruptible", return_value=True):
        input_control.tap_key_shifted("r", hold_time=0.05)

    assert mock_kb.press.call_args_list[0] == call("shift")
    assert mock_kb.press.call_args_list[1] == call("r")
    assert mock_kb.release.call_args_list[0] == call("r")
    assert mock_kb.release.call_args_list[1] == call("shift")


def test_tap_key_shifted_returns_interrupt_status():
    with patch("input_control.keyboard"), \
         patch("input_control.sleep_interruptible", return_value=False) as mock_sleep:
        result = input_control.tap_key_shifted("r", hold_time=0.05)
    assert result is False
