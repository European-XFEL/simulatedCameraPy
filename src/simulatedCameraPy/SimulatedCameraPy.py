#############################################################################
# Author: <andrea.parenti@xfel.eu>
# Created on November 12, 2013
# Copyright (C) European XFEL GmbH Schenefeld. All rights reserved.
#############################################################################

import random
import threading
import time

import numpy as np
import scipy.misc
import scipy.stats
from imageSource.CameraImageSource import CameraImageSource
from karabo.bound import (
    BOOL_ELEMENT, DOUBLE_ELEMENT, FLOAT_ELEMENT, INT32_ELEMENT,
    KARABO_CLASSINFO, NODE_ELEMENT, PATH_ELEMENT, SLOT_ELEMENT, STRING_ELEMENT,
    Encoding, Hash, State, Types, Unit, Worker
)

from ._version import version as deviceVersion

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


@KARABO_CLASSINFO("SimulatedCameraPy", deviceVersion)
class SimulatedCameraPy(CameraImageSource):
    def __init__(self, configuration):
        # always call PythonDevice constructor first!
        super(SimulatedCameraPy, self).__init__(configuration)

        # Define the first function to be called after the constructor has
        # finished
        self.registerInitialFunction(self.initialization)

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
        (
            SLOT_ELEMENT(expected).key("acquire")
            .displayedName("Acquire")
            .description("Start acquisition.")
            .allowedStates(State.ON)
            .commit(),

            SLOT_ELEMENT(expected).key("stop")
            .displayedName("Stop")
            .description("Stop acquisition.")
            .allowedStates(State.ACQUIRING)
            .commit(),

            SLOT_ELEMENT(expected).key("trigger")
            .displayedName("Trigger")
            .description("Sends a software trigger to the camera.")
            .allowedStates(State.ACQUIRING)
            .commit(),

            SLOT_ELEMENT(expected).key("reset")
            .displayedName("Reset")
            .description("Reset software error.")
            .allowedStates(State.ERROR)
            .commit(),

            BOOL_ELEMENT(expected).key("autoConnect")
            .displayedName("Auto Connect")
            .description("Auto-connect to the camera")
            .assignmentMandatory()
            .init()
            .commit(),

            DOUBLE_ELEMENT(expected).key("exposureTime")
            .displayedName("Exposure Time")
            .description("The requested exposure time in seconds")
            .unit(Unit.SECOND)
            .assignmentOptional().defaultValue(1.0)
            .minInc(0.02).maxInc(5.0)
            .reconfigurable()
            .commit(),

            INT32_ELEMENT(expected).key("pollInterval")
            .displayedName("Poll Interval")
            .description("The interval with which the camera should be polled")
            .unit(Unit.SECOND)
            .minInc(1)
            .assignmentOptional().defaultValue(10)
            .reconfigurable()
            .commit(),

            STRING_ELEMENT(expected).key("imageType")
            .displayedName("Image Type")
            .description("Select the simulated image type")
            .options("2d_Gaussian,RGB_Image,Grayscale_Image,"
                     "Load_from_file,FractalJulia")
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

            FLOAT_ELEMENT(expected).key("gaussian.jitter")
            .displayedName("Jitter")
            .description("The jitter on the gaussian position")
            .assignmentOptional().defaultValue(1)
            .minInc(1)
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
            if inputConfig.has('imageType'):
                self.image = None
                data = self.update_image()
                self.image = data
                self.updateOutputSchema()
                self.newImgAvailable = True

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

    def create_julia(self, n, m, c_real, c_imag,
                     p_scale=200, p_imax=512, p_thold=3.0):
        x = np.linspace(-m//p_scale, m//p_scale, num=m)
        y = np.linspace(-n//p_scale, n//p_scale, num=n)
        xg, yg = np.meshgrid(x, y)
        z_coord = xg+1j*yg

        c_const = (c_real+1j*c_imag) * np.ones([n, m])
        mask = np.full([n, m], True, dtype=np.bool)
        num = np.zeros([n, m])
        for itr in range(p_imax):
            z_coord[mask] = z_coord[mask]*z_coord[mask] + c_const[mask]
            mask[np.abs(z_coord) > p_thold] = False
            num[mask] = itr
        return num

    def initialization(self):
        self.log.INFO("SimulatedCameraPy.initialization")

        # register slots
        self.KARABO_SLOT(self.acquire)
        self.KARABO_SLOT(self.trigger)
        self.KARABO_SLOT(self.stop)
        self.KARABO_SLOT(self.reset)

        self.updateState(State.INIT)

        # Camera model
        self.set("cameraModel", "simCam")
        self.image = None
        data = self.update_image()
        self.image = data
        self.updateOutputSchema()
        self.newImgAvailable = True

        # Sensor geometry
        self.set("sensorHeight", data.shape[0])
        self.set("sensorWidth", data.shape[1])

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

        self.signal_eos()  # End-of-Stream signal

        self.set("cameraAcquiring", False)
        self.updateState(State.ON)

    def reset(self):
        self.log.INFO("SimulatedCameraPy.reset")
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
                    self.image = (self.image * newPixelGain).astype(
                        self.image.dtype)
                    pixelGain = newPixelGain
                    self.newImgAvailable = False

                # Prepare image
                data = self.update_image()
                self.image = data
                image_header = Hash(
                    "blockId", frames, "receptionTime", round(time.time()))

                # Write image to output channels
                self.write_channels(data, header=image_header)

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

    def update_image(self):
        """Updates the current image to simulate progress"""
        imageType = self.get("imageType")

        if imageType == '2d_Gaussian':
            width = self['gaussian.imageSizeX']
            height = self['gaussian.imageSizeY']
            jitter = self['gaussian.jitter']
            # Add some random noise
            im_noise = np.random.uniform(high=4000, size=[height, width])
            # Add a gaussian at random position
            im_beam = self.create_gaussian(
                self['gaussian.posX'] + int(
                    np.random.uniform(-jitter, jitter)),
                self['gaussian.posY'] + int(
                    np.random.uniform(-jitter, jitter)),
                self['gaussian.sigmaX'] * np.random.uniform(0.7, 1.2),
                self['gaussian.sigmaY'] * np.random.uniform(0.7, 1.2),
                width, height)
            image = im_beam + im_noise.astype('uint16')
        elif imageType == "FractalJulia":
            image = self.create_julia(800, 600,
                                      np.random.uniform(-1.0, 1.0),
                                      np.random.uniform(-1.0, 1.0)
                                      ).astype('uint16')
        elif self.image is None:
            # Special case when there's no previous image and we were loading
            # the image from file
            if imageType == 'Load_from_file':
                # Try to load image file
                filename = self.get("imageFilename")
                image = np.load(filename)
                self.log.INFO('Image loaded from file %s' % filename)
            elif imageType == 'RGB_Image':
                image = scipy.misc.face().astype('uint8')
                self.log.INFO('RGB image loaded')
            else:
                # Default image, grayscale, vertical gradient
                a = np.arange(500, dtype=np.uint16)
                b = np.array([a] * 1000)
                image = np.rot90(b)
                self.log.INFO('Default image (grayscale) loaded')
        else:
            # Roll image by 10 lines
            image = np.roll(self.image, 10, axis=0)
        return image

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
        shape = self.image.shape
        if self.image.ndim == 2:
            encoding = Encoding.GRAY
        elif self.image.ndim == 3:
            encoding = Encoding.RGB
        else:
            encoding = Encoding.UNDEFINED

        d_type = str(self.image.dtype)
        k_type = DTYPE_TO_KTYPE.get(d_type, None)

        self.update_output_schema(shape, encoding, k_type)
