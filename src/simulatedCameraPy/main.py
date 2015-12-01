#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="May, 2015, 04:01 PM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

from karabo.api_1 import Configurator, Hash
from SimulatedCameraPy import PythonDevice

if __name__ == "__main__":
    device = Configurator(PythonDevice).create("SimulatedCameraPy", Hash("Logger.priority", "DEBUG", "deviceId", "SimulatedCameraPyMain_0"))
    device.run()
