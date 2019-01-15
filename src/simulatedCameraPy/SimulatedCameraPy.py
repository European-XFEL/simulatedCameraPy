#!/usr/bin/env python

#############################################################################
# Author: <andrea.parenti@xfel.eu>
# Created on November 12, 2013
# Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
#############################################################################

import numpy as np
import random
import scipy.misc
import scipy.stats
import threading
import time

from karabo.bound import (
    BOOL_ELEMENT, CameraInterface, DaqDataType, DOUBLE_ELEMENT, FLOAT_ELEMENT,
    Hash, ImageData, IMAGEDATA_ELEMENT, INT32_ELEMENT, KARABO_CLASSINFO,
    NDARRAY_ELEMENT, NODE_ELEMENT, OUTPUT_CHANNEL, PATH_ELEMENT, PythonDevice,
    Schema, State, STRING_ELEMENT, Types, Unit, Worker
)

DTYPE_TO_KTYPE = {
    'uint8': Types.UINT8,
    'int8': Types.INT8,
    'uint16': Types.UINT16,
    'int16': Types.INT16,
    'uint32': Types.UINT32,
    'int32': Types.INT32,
    'uint64': Types.UINT64,
    'int64': Types.INT64,
    'float32': Types.FLOAT,
    'float': Types.DOUBLE,
    'double': Types.DOUBLE,
}


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
        self.newImgAvailable = False

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
                .assignmentOptional().defaultValue("2d_Gaussian")
                .init()
                .commit(),

            NODE_ELEMENT(expected).key("gaussian")
                .displayedName("Gaussian Parameters")
                .commit(),

            INT32_ELEMENT(expected).key("gaussian.imageSizeX")
                .displayedName("Image Width")
                .description("The size x of the image")
                .assignmentOptional().defaultValue(800)
                .minInc(800)
                .maxInc(1600)
                .reconfigurable()
                .commit(),

            INT32_ELEMENT(expected).key("gaussian.imageSizeY")
                .displayedName("Image Height")
                .description("The size y of the image")
                .assignmentOptional().defaultValue(600)
                .minInc(600)
                .maxInc(1200)
                .reconfigurable()
                .commit(),

            FLOAT_ELEMENT(expected).key("gaussian.posX")
                .displayedName("Position X")
                .description("The position x of the gaussian")
                .assignmentOptional().defaultValue(400)
                .minInc(0)
                .reconfigurable()
                .commit(),

            FLOAT_ELEMENT(expected).key("gaussian.posY")
                .displayedName("Position Y")
                .description("The position Y of the gaussian")
                .assignmentOptional().defaultValue(300)
                .minInc(0)
                .reconfigurable()
                .commit(),

            FLOAT_ELEMENT(expected).key("gaussian.sigmaX")
                .displayedName("Sigma X")
                .description("The sigma X of the gaussian")
                .assignmentOptional().defaultValue(100)
                .minInc(0)
                .reconfigurable()
                .commit(),

            FLOAT_ELEMENT(expected).key("gaussian.sigmaY")
                .displayedName("Sigma Y")
                .description("The sigma Y of the gaussian")
                .assignmentOptional().defaultValue(100)
                .minInc(0)
                .reconfigurable()
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

        if inputConfig.has('gaussian'):
            imageType = self['imageType']
            if inputConfig.has('imageType'):
                imageType = inputConfig['inputConfig']

            if imageType == '2d_Gaussian':
                pars = []
                for k in ['posX', 'posY', 'sigmaX', 'sigmaY', 'imageSizeX',
                          'imageSizeY']:
                    key = 'gaussian.' + k
                    if inputConfig.has(key):
                        pars.append(inputConfig[key])
                    else:
                        pars.append(self[key])

                # 2d Gaussian, no rotation
                # pos_x, sigma_x, pos_y, sigma_y, im_size_x, im_size_y
                data = self.create_gaussian(*pars)
                self.image = data
                self.updateOutputSchema()
                self.newImgAvailable = True
                self.log.INFO('Gaussian image updated')

                # Sensor geometry
                self.set("sensorHeight", data.shape[0])
                self.set("sensorWidth", data.shape[1])

    def create_gaussian(self, pos_x, pos_y, sigma_x, sigma_y,
                        im_size_x, im_size_y):
        rvx = scipy.stats.norm(pos_x, sigma_x)
        x = rvx.pdf(np.arange(im_size_x))  # 1d gaussian
        rvy = scipy.stats.norm(pos_y, sigma_y)
        y = rvy.pdf(np.arange(im_size_y))  # 1d gaussian
        z = np.outer(y, x)  # 2d gaussian (float64)
        # data -> uint16
        data = (z / z.max() * 1/2 * np.iinfo('uint16').max).astype('uint16')
        return data

    def initialize(self):
        self.log.INFO("SimulatedCameraPy.initialize")

        self.updateState(State.INIT)

        # Camera model
        self.set("cameraModel", "simCam")

        imageType = self.get("imageType")

        try:
            if imageType == "2d_Gaussian":
                # 2d Gaussian, no rotation
                # pos_x, sigma_x, pos_y, sigma_y, im_size_x, im_size_y
                data = self.create_gaussian(
                    self['gaussian.posX'], self['gaussian.posY'],
                    self['gaussian.sigmaX'], self['gaussian.sigmaY'],
                    self['gaussian.imageSizeX'], self['gaussian.imageSizeY'])
                self.log.INFO('Gaussian image created')
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
        self.updateOutputSchema()
        self.newImgAvailable = True

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

        pixelGain = None

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

                if self.newImgAvailable or newPixelGain != pixelGain:
                    # Copy original image and apply gain
                    image = (self.image * newPixelGain).astype(
                        self.image.dtype)
                    pixelGain = newPixelGain
                    self.newImgAvailable = False

                imageType = self.get("imageType")
                if imageType == '2d_Gaussian':
                    # Add some random noise
                    noise = np.random.uniform(high=4000, size=image.shape)
                    image2 = self.create_gaussian(
                        self['gaussian.posX']+int(np.random.uniform(-1.0,1.0)*100), self['gaussian.posY']+int(np.random.uniform(-1.0,1.0)*100),
                        self['gaussian.sigmaX']*np.random.uniform(0.7,1.2), self['gaussian.sigmaY']*np.random.uniform(0.7,1.2),
                        self['gaussian.imageSizeX'], self['gaussian.imageSizeY'])
                    image2 = image2 + noise.astype('uint16')
                else:
                    # Roll image by 10 lines
                    w = 10 * image.shape[0]
                    image = np.roll(image, w)
                    image2 = image

                # Write image via p2p
                print( "Writing channel ")
                imageData = ImageData(image2)
                imageData.setHeader(Hash("blockId", frames, "receptionTime",
                                         round(time.time())))
                self.writeChannel("output", Hash("data.image", imageData))

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

    def updateOutputSchema(self):
        # Get device configuration before schema update
        try:
            outputHostname = self["output.hostname"]
        except AttributeError as e:
            # Configuration does not contain "output.hostname"
            outputHostname = None

        shape = self.image.shape
        dType = str(self.image.dtype)
        kType = DTYPE_TO_KTYPE.get(dType, None)
        newSchema = Schema()
        outputData = Schema()
        (
            NODE_ELEMENT(outputData).key("data")
                .displayedName("Data")
                .setDaqDataType(DaqDataType.TRAIN)
                .commit(),

            IMAGEDATA_ELEMENT(outputData).key("data.image")
                .displayedName("Image")
                .setDimensions(str(shape).strip("()"))
                .commit(),

            # Set (overwrite) shape and dtype for internal NDArray element -
            # needed by DAQ
            NDARRAY_ELEMENT(outputData).key("data.image.pixels")
                .shape(str(shape).strip("()"))
                .dtype(kType)
                .commit(),

            # Set "maxSize" for vector properties - needed by DAQ
            outputData.setMaxSize("data.image.dims", len(shape)),
            outputData.setMaxSize("data.image.dimTypes", len(shape)),
            outputData.setMaxSize("data.image.roiOffsets", len(shape)),
            outputData.setMaxSize("data.image.binning", len(shape)),
            outputData.setMaxSize("data.image.pixels.shape", len(shape)),

            OUTPUT_CHANNEL(newSchema).key("output")
                .displayedName("Output")
                .dataSchema(outputData)
                .commit(),
        )

        self.updateSchema(newSchema)

        if outputHostname:
            # Restore configuration
            self.log.DEBUG("output.hostname: %s" % outputHostname)
            self.set("output.hostname", outputHostname)
