import time
from agent.android_controller import AndroidController

SERIAL = "emulator-5554"

ctrl = AndroidController(serial=SERIAL)
ctrl.home()
time.sleep(5)