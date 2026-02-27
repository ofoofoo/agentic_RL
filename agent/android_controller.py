"""
Android device controller via ppadb (pure-python-adb).

Avoids raw subprocess shell string construction — ppadb wraps the ADB
protocol in a typed Python client.
"""

import os
import time
from ppadb.client import Client as AdbClient
from PIL import Image, ImageDraw, ImageFont
import io


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

    # def screenshot(self, save_path: str) -> str:
    #     """
    #     Capture a screenshot and write it to *save_path* locally.
    #     ppadb returns the PNG bytes directly — no on-device tmp path needed.

    #     Returns the save path on success.
    #     """
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     png_bytes: bytes = self.device.screencap()
    #     with open(save_path, "wb") as f:
    #         f.write(png_bytes)
    #     return save_path

    def screen_size(self) -> tuple[int, int]:
        """
        Return the physical screen resolution as (width, height) in pixels.
        Parses the output of `adb shell wm size`, e.g. "Physical size: 1280x2856".
        """
        output: str = self.device.shell("wm size")
        # output looks like: "Physical size: 1280x2856\n"
        for line in output.splitlines():
            if "Physical size" in line or "Override size" in line:
                _, _, dimensions = line.partition(":")
                w, _, h = dimensions.strip().partition("x")
                return int(w), int(h)
        raise RuntimeError(f"Could not parse screen size from: {output!r}")
    
    def screenshot_with_numbered_grid(
        self,
        save_path: str,
        grid_path: str,
    ) -> tuple[str, int, int, float, float]:
        """
        Capture a screenshot and annotate it with a numbered cell grid.

        Returns (grid_path, rows, cols, t_adb_s, t_preprocess_s).
          - t_adb_s:        time for ADB screencap (phone → Mac transfer)
          - t_preprocess_s: time for PIL decode + grid drawing + disk write
        """
        CELL_W, CELL_H = 80, 119
        cols = 1280 // CELL_W   # 16
        rows = 2856 // CELL_H   # 24

        # ── ADB capture ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        png_bytes: bytes = self.device.screencap()
        t_adb = time.perf_counter() - t0

        # ── Image preprocessing (decode → draw grid → save) ───────────
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        draw = ImageDraw.Draw(img)
        color = (255, 116, 113)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=25)

        for r in range(rows):
            for c in range(cols):
                label = r * cols + c + 1
                x0, y0 = c * CELL_W, r * CELL_H
                x1, y1 = x0 + CELL_W, y0 + CELL_H
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0 + 4, y0 + 4), str(label), fill=color, font=font)

        os.makedirs(os.path.dirname(grid_path) or ".", exist_ok=True)
        img.save(grid_path)
        t_preprocess = time.perf_counter() - t0

        return grid_path, rows, cols, t_adb, t_preprocess



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
