#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="November, 2013, 11:22 AM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

from karabo.configurator import Configurator
from PythonSimulatedCamera import *

if __name__ == "__main__":
    device = Configurator(PythonDevice).create("PythonSimulatedCamera", Hash("Logger.priority", "DEBUG", "deviceId", "PythonSimulatedCameraMain_0"))
    device.run()
