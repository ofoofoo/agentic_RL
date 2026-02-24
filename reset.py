import time
from agent.android_controller import AndroidController

SERIAL = "emulator-5554"

ctrl = AndroidController(serial=SERIAL)
ctrl.home()
print("=====Home button pressed, restring for 5 seconds.=====")
time.sleep(5)