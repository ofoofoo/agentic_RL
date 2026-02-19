
import time
from agent.android_controller import AndroidController

SERIAL = "emulator-5554"

ctrl = AndroidController(serial=SERIAL)
w, h = ctrl.screen_size()
print(f"Connected to {SERIAL}  |  screen: {w}x{h}")

ctrl.screenshot("/tmp/test_shot.png")
print("✓ Screenshot saved to /tmp/test_shot.png")
time.sleep(1)

print("tapping 442")
ctrl.tap(442, 698)
time.sleep(10)
ctrl.tap(1075, 2358.8)
time.sleep(10)

ctrl.home()
print("went to home")
time.sleep(10)

x_mid = w // 2
y_start = int(h * 0.70)
y_end   = int(h * 0.30)
print(f"Swiping up: ({x_mid},{y_start}) -> ({x_mid},{y_end})")
ctrl.swipe(x_mid, y_start, x_mid, y_end, duration_ms=500)