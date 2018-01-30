#!/usr/bin/env python

#############################################################################
# Author: <andrea.parenti@xfel.eu>
# Created on November 12, 2013
# Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
#############################################################################

import numpy as np
import os
import random
import scipy.misc
import scipy.stats
import threading
import time

from karabo.bound import (
    BOOL_ELEMENT, DOUBLE_ELEMENT, INT32_ELEMENT, KARABO_CLASSINFO,
    PATH_ELEMENT, STRING_ELEMENT, CameraInterface, Hash, ImageData,
    PythonDevice, State, Unit, Worker
)


@KARABO_CLASSINFO("SimulatedCameraPy", "2.2")
class SimulatedCameraPy(PythonDevice, CameraInterface):
    def __init__(self, configuration):
        # always call PythonDevice constructor first!
        super(SimulatedCameraPy, self).__init__(configuration)

        random.seed()

        self.pollWorker = None

        self.keepAcquiring = False
        self.swTriggerReceived = False

        # Sample Image
        self.image = None

        self.filePath = ""
        self.fileName = ""
        self.fileType = ""
        self.fileCounter = 0

        # Condition variable and lock, to wake up acquireImages()
        # when trigger is received
        self.condLock = threading.Lock()
        self.condVar = threading.Condition(self.condLock)
        # Acquire thread
        self.acquireThread = None

    @staticmethod
    def expectedParameters(expected):
        '''Description of device parameters statically known'''
        (
            BOOL_ELEMENT(expected).key("autoConnect")
                .displayedName("Auto Connect")
                .description("Auto-connect to the camera")
                .assignmentMandatory()
                .init()
                .commit(),

            STRING_ELEMENT(expected).key("imageType")
                .displayedName("Image Type")
                .description("Select the simulated image type")
                .options("2d_Gaussian,RGB_Image,Grayscale_Image,"
                         "Load_from_file")
                .assignmentOptional().defaultValue("Load_from_file")
                .init()
                .commit(),

            PATH_ELEMENT(expected).key("imageFilename")
                .displayedName("Image Filename")
                .description("The full filename to the fake image displayed "
                             "by the camera. File format must be 'npy'.")
                .assignmentOptional()
                .defaultValue("european-xfel-logo-greyscales.npy")
                .init()
                .commit(),

            DOUBLE_ELEMENT(expected).key("pixelGain")
                .displayedName("Pixel Gain")
                .description("The pixel gain")
                .assignmentOptional().defaultValue(0.5)
                .minInc(0.001)
                .maxInc(1.000)
                .reconfigurable()
                .commit(),

            STRING_ELEMENT(expected).key("cycleMode")
                .displayedName("Cycle Mode")
                .description("Configures whether the camera will acquire a "
                             "fixed length sequence or a continuous sequence")
                .assignmentOptional().defaultValue("Continuous")
                .options("Fixed Continuous")
                .reconfigurable()
                .allowedStates(State.ON)
                .commit(),

            INT32_ELEMENT(expected).key("frameCount")
                .displayedName("Frame Count")
                .description("Configures the number of images to acquire in "
                             "the sequence, when the camera is in 'Fixed' "
                             "Mode")
                .assignmentOptional().defaultValue(1)
                .reconfigurable()
                .allowedStates(State.ON)
                .commit(),

            STRING_ELEMENT(expected).key("triggerMode")
                .displayedName("Trigger Mode")
                .description("Allows the user to configure the camera trigger"
                             " mode at a high level")
                .assignmentOptional().defaultValue("Internal")
                .options("Internal Software")
                .reconfigurable()
                .allowedStates(State.ON)
                .commit(),

            ###################################
            #  READ ONLY HARDWARE PARAMETERS  #
            ###################################
            DOUBLE_ELEMENT(expected).key("sensorTemperature")
                .displayedName("Sensor Temperature")
                .description("Returns the temperature of the sensor in "
                             "Celsius degrees")
                .unit(Unit.DEGREE_CELSIUS)
                .readOnly()
                .commit(),

            INT32_ELEMENT(expected).key("sensorWidth")
                .displayedName("Sensor Width")
                .description("Returns the width of the sensor in pixels")
                .readOnly()
                .commit(),

            INT32_ELEMENT(expected).key("sensorHeight")
                .displayedName("Sensor Height")
                .description("Returns the height of the sensor in pixels")
                .readOnly()
                .commit(),

            BOOL_ELEMENT(expected).key("cameraAcquiring")
                .displayedName("Camera Acquiring")
                .description("Returns whether or not an acquisition is "
                             "currently running")
                .readOnly()
                .commit(),

            STRING_ELEMENT(expected).key("cameraModel")
                .displayedName("Camera Model")
                .description("Returns the camera model")
                .readOnly()
                .commit(),
        )

    def preReconfigure(self, inputConfig):
        self.log.INFO("SimulatedCameraPy.preReconfigure")
        if inputConfig.has("pollInterval") and self.pollWorker is not None:
            timeout = 1000 * inputConfig.get("pollInterval")  # to milliseconds
            self.pollWorker.setTimeout(timeout)

    def initialize(self):
        self.log.INFO("SimulatedCameraPy.initialize")

        self.updateState(State.INIT)

        # Camera model
        self.set("cameraModel", "simCam")

        imageType = self.get("imageType")

        try:
            if imageType == "2d_Gaussian":
                # 2d Gaussian, no rotation
                rvx = scipy.stats.norm(300, 50)
                x = rvx.pdf(np.arange(800))  # 1d gaussian
                rvy = scipy.stats.norm(200, 75)
                y = rvy.pdf(np.arange(600))  # 1d gaussian
                z = np.outer(y, x)  # 2d gaussian (float64)
                data = (
                    z / z.max() * 0.5 * np.iinfo('uint16').max
                ).astype('uint16')  # -> uint16
                self.log.INFO('2d gaussian image loaded')
            elif imageType == 'RGB_Image':
                # RGB image
                red = [255, 0, 0]
                green = [0, 255, 0]
                blue = [0, 0, 255]
                data = np.append(
                    np.append([red * 67500], [green * 67500]), [blue * 67500]
                ).reshape(450, 450, 3).astype('uint16')
                self.log.INFO('RGB image loaded')
            elif imageType == 'Load_from_file':
                # Try to load image file
                filename = self.get("imageFilename")
                data = np.load(filename)
                self.log.INFO('Image loaded from file %s' % filename)
            else:
                # Default image, grayscale, vertical gradient
                a = np.arange(500, dtype=np.uint16)
                b = np.array([a] * 1000)
                data = np.rot90(b)
                self.log.INFO('Default image (grayscale) loaded')
        except Exception as e:
            # Default image, grayscale, vertical gradient
            a = np.arange(500, dtype=np.uint16)
            b = np.array([a] * 1000)
            data = np.rot90(b)
            self.log.WARN(str(e))
            self.log.INFO('Default image (grayscale) loaded')

        self.image = data

        # Sensor geometry
        self.set("sensorHeight", data.shape[0])
        self.set("sensorWidth", data.shape[1])

        if self.pollWorker is None:
            # Create and start poll worker
            timeout = 1000 * self.get("pollInterval")  # to milliseconds
            self.pollWorker = Worker(self.pollHardware, timeout, -1)
            self.pollWorker.daemon = True
            self.pollWorker.start()

        # Sleep a while (to simulate camera initialization)
        time.sleep(1)

        # Change state, depending on the "autoConnect" option
        autoConnect = self.get("autoConnect")
        if autoConnect:
            self.updateState(State.ON)
        else:
            self.updateState(State.UNKNOWN)

    def connectCamera(self):
        self.log.INFO("SimulatedCameraPy.connectCamera")
        self.updateState(State.ON)

    def acquire(self):
        self.log.INFO("SimulatedCameraPy.acquire")

        # Get file path, name and extension
        newName = False
        if self.get("imageStorage.filePath") != self.filePath:
            # New file path
            self.filePath = self.get("imageStorage.filePath")
            newName = True
        if self.get("imageStorage.fileName") != self.fileName:
            # New file root name
            self.fileName = self.get("imageStorage.fileName")
            newName = True
        if self.get("imageStorage.fileType") != self.fileType:
            # New file extension
            self.fileType = self.get("imageStorage.fileType")

        if newName:
            # File has changed name... reset counter
            self.fileCounter = 0

        # Start acquire thread, since slots cannot block
        self.keepAcquiring = True
        self.swTriggerReceived = False
        self.acquireThread = threading.Thread(target=self.acquireImages)
        self.acquireThread.daemon = True
        self.acquireThread.start()

        # Change state
        self.updateState(State.ACQUIRING)
        self.set("cameraAcquiring", True)

    def trigger(self):
        self.log.INFO("SimulatedCameraPy.trigger")

        # Will notify acquireImages to continue
        self.condVar.acquire()
        self.swTriggerReceived = True
        self.condVar.notify_all()
        self.condVar.release()

    def stop(self):
        self.log.INFO("SimulatedCameraPy.stop")

        self.keepAcquiring = False  # Signal acquire thread to quit

        # If running with software trigger,
        # must notify acquire thread to continue
        self.condVar.acquire()
        self.swTriggerReceived = False
        self.condVar.notify_all()
        self.condVar.release()

        # Wait for acquire thread to join
        if self.acquireThread is not None and self.acquireThread.isAlive():
            self.acquireThread.join(10.)

        # Signals end of stream
        self.signalEndOfStream("output")

        self.set("cameraAcquiring", False)
        self.updateState(State.ON)

    def resetHardware(self):
        self.log.INFO("SimulatedCameraPy.resetHardware")
        self.updateState(State.ON)

    def pollHardware(self):
        self.log.DEBUG("SimulatedCameraPy.pollHardware")
        temperature = 25.4 + random.random() / 10.
        self.set("sensorTemperature", temperature)

    def acquireImages(self):
        self.log.INFO("SimulatedCameraPy.acquireImages")

        # Is software trigger?
        triggerModeIsSoftware = False
        if self.get("triggerMode") == "Software":
            triggerModeIsSoftware = True

        # Is cycle mode "fixed"?
        cycleModeIsFixed = False
        if self.get("cycleMode") == "Fixed":
            cycleModeIsFixed = True
        # How many frames should be acquired
        frameCount = self.get("frameCount")
        # Frame counter
        frames = 0

        saveImages = self.get("imageStorage.enable")
        pixelGain = self.get("pixelGain")
        # Copy original image and apply gain
        image = (self.image * pixelGain).astype(self.image.dtype)

        while self.keepAcquiring:

            try:
                if triggerModeIsSoftware:
                    # Running in SW trigger mode

                    # Wait notification
                    self.condVar.acquire()
                    self.condVar.wait()
                    self.condVar.release()

                    if not self.swTriggerReceived:
                        # No sw trigger -> continue
                        continue
                    else:
                        self.swTriggerReceived = False

                exposureTime = self.get("exposureTime")
                newPixelGain = self.get("pixelGain")

                # Sleep for "exposureTime" to simulate image acquisition
                time.sleep(exposureTime)

                # Pixel gain has changed
                if newPixelGain != pixelGain:
                    image *= newPixelGain / pixelGain
                    pixelGain = newPixelGain

                imageType = self.get("imageType")
                if imageType == '2d_Gaussian':
                    # Add some random noise
                    noise = np.random.uniform(high=20, size=image.shape)
                    image2 = image + noise.astype('uint8')
                else:
                    # Roll image by 10 lines
                    w = 10 * image.shape[0]
                    image = np.roll(image, w)
                    image2 = image

                # Write image via p2p
                imageData = ImageData(image2)
                imageData.setHeader(Hash("blockId", frames, "receptionTime",
                                         round(time.time())))
                self.writeChannel("output", Hash("image", imageData))

                if saveImages:
                    # Create filename (without path and extension)
                    imgname = "%s%06ld" % (self.fileName, self.fileCounter)
                    # Prepend path and append extension
                    imgname = os.path.join(self.filePath, "{}.{}"
                                           .format(imgname, self.fileType))

                    if self.fileType in ["tif", "jpg", "png"]:
                        # PIL must installed!
                        scipy.misc.imsave(imgname, image2)
                    else:
                        raise ValueError("File type not supported")

                    self.set("imageStorage.lastSaved", imgname)
                    self.fileCounter += 1

                frames += 1
                if cycleModeIsFixed and frames >= frameCount:
                    # change state, quit loop
                    self.set("cameraAcquiring", False)
                    self.updateState(State.ON)
                    break

            except Exception as e:
                # log error, change state, quit loop
                self.log.ERROR("SimulatedCameraPy.acquireImages: %s" % str(e))
                self.set("cameraAcquiring", False)
                self.updateState(State.ERROR)
                break

    def preDestruction(self):
        # Stop polling camera
        if self.pollWorker:
            if self.pollWorker.is_running():
                self.pollWorker.stop()
            self.pollWorker.join()

        # Stop acquisition, if running
        if self.get('state') == State.ACQUIRING:
            self.execute("stop")
