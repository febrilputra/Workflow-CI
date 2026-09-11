"""Regression checks for real startup failures observed on the CI runner."""
import http.client
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_artifacts


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class StartupHealthTests(unittest.TestCase):
    def test_transient_reset_or_bad_http_response_can_recover(self):
        for startup_error in (ConnectionResetError(104, "Connection reset by peer"),
                              http.client.BadStatusLine("server still starting")):
            with self.subTest(error=type(startup_error).__name__):
                clock = FakeClock()
                response = MagicMock()
                response.__enter__.return_value.status = 200
                with ExitStack() as stack:
                    stack.enter_context(patch.object(ci_artifacts.time, "monotonic", clock.monotonic))
                    stack.enter_context(patch.object(ci_artifacts.time, "sleep", clock.sleep))
                    opener = stack.enter_context(patch.object(ci_artifacts.urllib.request, "urlopen",
                                                              side_effect=[startup_error, response]))
                    ci_artifacts.wait_for_health("http://model:8080")
                self.assertEqual(opener.call_count, 2)
                self.assertGreater(clock.now, 0)
                self.assertLess(clock.now, 180)

    def test_repeated_transport_failures_expire_at_deadline(self):
        clock = FakeClock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(ci_artifacts.time, "monotonic", clock.monotonic))
            stack.enter_context(patch.object(ci_artifacts.time, "sleep", clock.sleep))
            opener = stack.enter_context(patch.object(ci_artifacts.urllib.request, "urlopen",
                                                      side_effect=ConnectionResetError(104, "not ready")))
            with self.assertRaisesRegex(RuntimeError, "within 5 seconds"):
                ci_artifacts.wait_for_health("http://model:8080", timeout_seconds=5)
        self.assertEqual(clock.now, 5)
        self.assertGreater(opener.call_count, 1)


if __name__ == "__main__":
    unittest.main()
