"""
Android device controller via ppadb (pure-python-adb).

Avoids raw subprocess shell string construction — ppadb wraps the ADB
protocol in a typed Python client.
"""

import os
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
    
    def screenshot_with_grid(
        self,
        save_path: str,
        grid_path: str,
        step: int = 200,
    ) -> str:
        """
        Take a screenshot, then save a copy annotated with a coordinate grid
        to *grid_path*. The grid lines and labels are spaced every *step* pixels,
        giving the vision model clear spatial anchors.

        Returns grid_path.
        """
        png_bytes: bytes = self.device.screencap()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=40)

        line_color = (255, 80, 80)    # red-ish grid lines
        label_color = (255, 80, 80)   # yellow labels (visible on most backgrounds)

        # Vertical lines + x labels
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=line_color, width=1)
            draw.text((x + 3, 4), str(x), fill=label_color, font=font)

        # Horizontal lines + y labels
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=line_color, width=1)
            draw.text((4, y + 3), str(y), fill=label_color, font=font)

        os.makedirs(os.path.dirname(grid_path) or ".", exist_ok=True)
        img.save(grid_path)
        return grid_path



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
