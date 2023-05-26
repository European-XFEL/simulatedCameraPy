import unittest

from karabo.bound import Configurator, Hash, PythonDevice

from ..SimulatedCameraPy import SimulatedCameraPy


class SimulatedCamera_TestCase(unittest.TestCase):
    def test_simulated_camera(self):
        config = Hash("Logger.priority", "WARN",
                      "deviceId", "SimulatedCamera_0",
                      "autoConnect", True)
        cam = Configurator(PythonDevice).create("SimulatedCameraPy", config)
        cam.startFsm()

    def test_simulated_camera_no_autoconnect(self):
        try:
            config = Hash("Logger.priority", "WARN", "deviceId",
                          "SimulatedCamera_noAutoConnect_0")
            cam = Configurator(PythonDevice).create("SimulatedCameraPy",
                                                    config)
            cam.startFsm()
            raise RuntimeError("Should have failed - no autoConnect provided")
        except:
            # OK
            pass


if __name__ == '__main__':
    unittest.main()
