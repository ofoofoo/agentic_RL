from __future__ import annotations
"""
Android device controller via ppadb (pure-python-adb).
"""

import os
import io
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from ppadb.client import Client as AdbClient
from PIL import Image, ImageDraw, ImageFont

MIN_DIST = 30

@dataclass
class UIElement:
    """One interactive element extracted from a uiautomator XML dump."""
    uid: str                        # resource-id or synthetic id
    bbox: tuple                     # ((x1, y1), (x2, y2))
    center: tuple = (0, 0)         # (cx, cy)
    attrib: str = "clickable"      # "clickable" or "focusable"
    text: str = ""                 # android:text
    content_desc: str = ""         # content-description


def _get_id_from_element(elem) -> str:
    bounds = elem.attrib["bounds"][1:-1].split("][")
    x1, y1 = map(int, bounds[0].split(","))
    x2, y2 = map(int, bounds[1].split(","))
    elem_w, elem_h = x2 - x1, y2 - y1

    if elem.attrib.get("resource-id"):
        elem_id = elem.attrib["resource-id"].replace(":", ".").replace("/", "_")
    else:
        elem_id = f"{elem.attrib.get('class', 'View')}_{elem_w}_{elem_h}"

    cd = elem.attrib.get("content-desc", "")
    if cd and len(cd) < 20:
        elem_id += f"_{cd.replace('/', '_').replace(' ', '').replace(':', '_')}"
    return elem_id


def _traverse_tree(xml_path: str, attrib: str, add_index: bool = True) -> list[UIElement]:
    """
    Parse a uiautomator XML dump and return clickable elements
    """
    elem_list: list[UIElement] = []
    path_stack = []
    for event, elem in ET.iterparse(xml_path, ["start", "end"]):
        if event == "start":
            path_stack.append(elem)
            if elem.attrib.get(attrib) != "true":
                continue
            try:
                bounds = elem.attrib["bounds"][1:-1].split("][")
                x1, y1 = map(int, bounds[0].split(","))
                x2, y2 = map(int, bounds[1].split(","))
            except (KeyError, ValueError):
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # skip if too close to an existing element
            close = False
            for e in elem_list:
                ecx, ecy = e.center
                dist = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
                if dist <= MIN_DIST:
                    close = True
                    break
            if close:
                continue

            uid = _get_id_from_element(elem)
            if add_index:
                uid += f"_{elem.attrib.get('index', '0')}"
            # parent prefix
            if len(path_stack) > 1:
                uid = _get_id_from_element(path_stack[-2]) + "_" + uid

            elem_list.append(UIElement(
                uid=uid,
                bbox=((x1, y1), (x2, y2)),
                center=(cx, cy),
                attrib=attrib,
                text=elem.attrib.get("text", ""),
                content_desc=elem.attrib.get("content-desc", ""),
            ))
        elif event == "end":
            if path_stack:
                path_stack.pop()
    return elem_list


class AndroidController:
    def __init__(self, serial: str, host: str = "127.0.0.1", port: int = 5037):
        """
        Connect to the ADB server and select a device by serial.
        Make sure `adb start-server` has been run (Android Studio does this
        automatically when you launch an emulator).
        """
        client = AdbClient(host=host, port=port)
        self.device = client.device(serial)
        if self.device is None:
            raise RuntimeError(
                f"Device '{serial}' not found. "
                f"Run `adb devices` to check connected devices."
            )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def screen_size(self) -> tuple[int, int]:
        """
        Return the physical screen resolution as (width, height) in pixels.
        Parses the output of `adb shell wm size`.
        """
        output: str = self.device.shell("wm size")
        for line in output.splitlines():
            if "Physical size" in line or "Override size" in line:
                _, _, dimensions = line.partition(":")
                w, _, h = dimensions.strip().partition("x")
                return int(w), int(h)
        raise RuntimeError(f"Could not parse screen size from: {output!r}")

    # ------------------------------------------------------------------
    # Mode 1 — UI hierarchy (primary)
    # ------------------------------------------------------------------

    def get_ui_hierarchy(self, xml_save_path: str) -> str:
        """
        Dump the UI hierarchy via uiautomator and pull the XML to *xml_save_path*.
        Returns the local path on success.
        """
        device_xml = "/sdcard/ui_dump.xml"
        self.device.shell(f"uiautomator dump --compressed {device_xml}")
        os.makedirs(os.path.dirname(xml_save_path) or ".", exist_ok=True)
        # pull via shell cat (ppadb doesn't have pull)
        xml_bytes = self.device.shell(f"cat {device_xml}")
        with open(xml_save_path, "w") as f:
            f.write(xml_bytes)
        return xml_save_path

    @staticmethod
    def parse_ui_elements(xml_path: str) -> list[UIElement]:
        """
        Parse a uiautomator XML dump and return a merged, de-duplicated list
        of clickable + focusable elements (clickable first).
        """
        clickable = _traverse_tree(xml_path, "clickable", add_index=True)
        focusable = _traverse_tree(xml_path, "focusable", add_index=True)

        merged = list(clickable)
        for fe in focusable:
            close = False
            for ce in clickable:
                dist = (
                    (fe.center[0] - ce.center[0]) ** 2
                    + (fe.center[1] - ce.center[1]) ** 2
                ) ** 0.5
                if dist <= MIN_DIST:
                    close = True
                    break
            if not close:
                merged.append(fe)
        return merged

    def screenshot_with_elements(
        self,
        labeled_path: str,
        xml_save_path: str | None = None,
    ) -> tuple[str, list[UIElement], float, float, float]:
        """
        Capture a screenshot, dump the UI hierarchy, label each interactive
        element with a number on the image.

        Returns (labeled_path, elem_list, t_adb_s, t_hierarchy_s, t_label_s).
        """
        # ── ADB screenshot ──────────────────────────────────────────────
        t0 = time.perf_counter()
        png_bytes: bytes = self.device.screencap()
        t_adb = time.perf_counter() - t0

        # ── UI hierarchy dump + parse ───────────────────────────────────
        t0 = time.perf_counter()
        if xml_save_path is None:
            xml_save_path = labeled_path.replace(".png", ".xml")
        self.get_ui_hierarchy(xml_save_path)
        elem_list = self.parse_ui_elements(xml_save_path)
        t_hierarchy = time.perf_counter() - t0

        # ── Draw labels on screenshot ───────────────────────────────────
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=28)
        except OSError:
            font = ImageFont.load_default()

        for idx, elem in enumerate(elem_list, 1):
            (x1, y1), (x2, y2) = elem.bbox
            # draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 116, 113), width=3)
            # draw label at center
            cx, cy = elem.center
            label = str(idx)
            # background rectangle for readability
            tw = len(label) * 14 + 8
            th = 28
            lx = cx - tw // 2
            ly = cy - th // 2
            draw.rectangle([lx, ly, lx + tw, ly + th], fill=(0, 0, 0, 180))
            draw.text((lx + 4, ly + 2), label, fill=(255, 116, 113), font=font)

        os.makedirs(os.path.dirname(labeled_path) or ".", exist_ok=True)
        img.save(labeled_path)
        t_label = time.perf_counter() - t0

        return labeled_path, elem_list, t_adb, t_hierarchy, t_label

    # ------------------------------------------------------------------
    # Mode 2 — numbered grid (fallback)
    # ------------------------------------------------------------------

    def screenshot_with_numbered_grid(
        self,
        save_path: str,
        grid_path: str,
    ) -> tuple[str, int, int, float, float]:
        """
        Capture a screenshot and annotate it with a numbered cell grid.

        Returns (grid_path, rows, cols, t_adb_s, t_preprocess_s).
        """
        CELL_W, CELL_H = 80, 119
        cols = 1280 // CELL_W   # 16
        rows = 2856 // CELL_H   # 24

        t0 = time.perf_counter()
        png_bytes: bytes = self.device.screencap()
        t_adb = time.perf_counter() - t0

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

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        self.device.input_swipe(x, y, x, y, duration_ms)

    def type_text(self, text: str) -> None:
        self.device.input_text(text)

    def clear_text(self) -> None:
        """Select all text in the focused field and delete it."""
        self.device.input_keycombination("113 29")
        self.device.input_keyevent("67")

    def enter(self) -> None:
        """Press the enter key."""
        self.device.input_keyevent("KEYCODE_ENTER")

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
