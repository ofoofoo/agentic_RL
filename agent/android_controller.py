"""
Android device controller via ppadb (pure-python-adb).

Avoids raw subprocess shell string construction — ppadb wraps the ADB
protocol in a typed Python client.
"""

import os
from ppadb.client import Client as AdbClient


class AndroidController:
    def __init__(self, serial: str, host: str = "127.0.0.1", port: int = 5037):
        """
        Connect to the ADB server and select a device by serial.
        Make sure `adb start-server` has been run (Android Studio does this
        automatically when you launch an emulator).

        Args:
            serial: Device serial, e.g. "emulator-5554". Run `adb devices` to list.
        """
        client = AdbClient(host=host, port=port)
        self.device = client.device(serial)
        if self.device is None:
            raise RuntimeError(
                f"Device '{serial}' not found. "
                f"Run `adb devices` to check connected devices."
            )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def screenshot(self, save_path: str) -> str:
        """
        Capture a screenshot and write it to *save_path* locally.
        ppadb returns the PNG bytes directly — no on-device tmp path needed.

        Returns the save path on success.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        png_bytes: bytes = self.device.screencap()
        with open(save_path, "wb") as f:
            f.write(png_bytes)
        return save_path

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def tap(self, x: int, y: int) -> None:
        self.device.input_tap(x, y)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 400,
    ) -> None:
        self.device.input_swipe(x1, y1, x2, y2, duration_ms)

    def type_text(self, text: str) -> None:
        # ppadb handles spaces and special chars more robustly than raw adb text
        self.device.input_text(text)

    def back(self) -> None:
        self.device.input_keyevent("KEYCODE_BACK")

    def home(self) -> None:
        self.device.input_keyevent("KEYCODE_HOME")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices(host: str = "127.0.0.1", port: int = 5037) -> list[str]:
        """Return a list of connected device serials."""
        client = AdbClient(host=host, port=port)
        return [d.serial for d in client.devices()]
