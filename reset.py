import time
from agent.android_controller import AndroidController

SERIAL = "emulator-5554"

ctrl = AndroidController(serial=SERIAL)
# ctrl.home()
# test delete text
ctrl.device.shell("input keycombination 113 29 && input keyevent 67")
# ctrl.device.shell("input text 'hello world'")
# ctrl.device.shell("input keyevent KEYCODE_MOVE_END")
# for _ in range(250):
#     ctrl.device.shell("input keyevent KEYCODE_DEL")
print("=====Home button pressed, restring for 5 seconds.=====")
time.sleep(5)