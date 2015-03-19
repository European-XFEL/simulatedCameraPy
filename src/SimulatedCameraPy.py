#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="November 12, 2013, 11:22 AM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

import time
import threading
from karabo.device import *
from karabo.camera_fsm import CameraFsm

import numpy
import scipy.stats
import random

@KARABO_CLASSINFO("SimulatedCameraPy", "1.0 1.1 1.2")
class SimulatedCameraPy(PythonDevice, CameraFsm):

    def __init__(self, configuration):
        # always call PythonDevice constructor first!
        super(SimulatedCameraPy,self).__init__(configuration)
        
        self.output = self._ss.createOutputChannelRawImageData("output", configuration)
        
        random.seed()
        
        self.doAcquire = False
        self.triggerReceived = False
        self.acqThread = None
        
        # Prepares and starts the "always run" thread
        self.doAlwaysRun = True
        self.alwaysRunThread = threading.Thread(target = self.alwaysRun)
        self.alwaysRunThread.start()
        
        # Prepares (but does not start) the polling thread
        self.doPolling  = True
        self.pollThread = threading.Thread(target = self.pollHardware)

        # Image
        self.image = None
    
    def __del__(self):
        self.output = None
        super(SimulatedCameraPy, self).__del__()
    
    @staticmethod
    def expectedParameters(expected):
        '''Description of device parameters statically known'''
        (
        
        OUTPUT_ELEMENT(expected).key("output")
                .displayedName("Output")
                .description("Output")        
                .setOutputType(OutputRawImageData)
                .commit()
        ,
        IMAGE_ELEMENT(expected).key("image")
                .displayedName("Image")
                .description("Image")
                .commit()
        ,
        STRING_ELEMENT(expected).key("imageType")
                .displayedName("Image Type")
                .description("Select the simulated image type")
                .options("2d_Gaussian,RGB_Image,Grayscale_Image,Load_from_file")
                .assignmentOptional().defaultValue("Load_from_file")
                .init()
                .commit()
        ,
        PATH_ELEMENT(expected).key("imageFilename")
                .displayedName("Image Filename")
                .description("The full filename to the fake image displayed by the camera. File format must be 'npy'.")
                .assignmentOptional().defaultValue("european-xfel-logo-greyscales.npy")
                .init()
                .commit()
        ,
        DOUBLE_ELEMENT(expected).key("exposureTime")
                .displayedName("Exposure Time")
                .description("The requested exposure time in seconds")
                .unit(Unit.SECOND)
                .assignmentOptional().defaultValue(1.0)
                .minInc(0.02)
                .maxInc(5.0)
                .reconfigurable()
                .commit()
        ,
        DOUBLE_ELEMENT(expected).key("pixelGain")
                .displayedName("Pixel Gain")
                .description("The pixel gain")
                .assignmentOptional().defaultValue(0.5)
                .minInc(0.001)
                .maxInc(1.000)
                .reconfigurable()
                .commit()
        ,
        STRING_ELEMENT(expected).key("cycleMode")
                .displayedName("Cycle Mode")
                .description("Configures whether the camera will acquire a fixed length sequence or a continuous sequence")
                .assignmentOptional().defaultValue("Continuous")
                .options("Fixed Continuous")
                .reconfigurable()
                .allowedStates("Ok.Ready")
                .commit()
        ,
        INT32_ELEMENT(expected).key("frameCount")
                .displayedName("Frame Count")
                .description("Configures the number of images to acquire in the sequence,"
                             "when the camera is in 'Fixed' Mode")
                .assignmentOptional().defaultValue(1)
                .reconfigurable()
                .allowedStates("Ok.Ready")
                .commit()
        ,
        STRING_ELEMENT(expected).key("triggerMode")
                .displayedName("Trigger Mode")
                .description("Allows the user to configure the camera trigger mode at a high level")
                .assignmentOptional().defaultValue("Internal")
                .options("Internal Software")
                .reconfigurable()
                .allowedStates("Ok.Ready")
                .commit()
        ,
        INT32_ELEMENT(expected).key('pollingInterval')
                .displayedName("Polling Interval")
                .description("The interval for polling the camera for read-out values.")
                .assignmentOptional().defaultValue(2)
                .minInc(1)
                .maxInc(60)
                .unit(Unit.SECOND)
                .init()
                .commit()
        ,
        
        ###################################
        #  READ ONLY HARDWARE PARAMETERS  #
        ###################################
                DOUBLE_ELEMENT(expected).key("sensorTemperature")
                .displayedName("Sensor Temperature")
                .description("Returns the temperature of the sensor in Celsius degrees")
                .unit(Unit.DEGREE_CELSIUS)
                .readOnly()
                .commit()
        ,
        INT32_ELEMENT(expected).key("sensorHeight")
                .displayedName("Sensor Height")
                .description("Returns the height of the sensor in pixels")
                .readOnly()
                .commit()
        ,
        INT32_ELEMENT(expected).key("sensorWidth")
                .displayedName("Sensor Width")
                .description("Returns the width of the sensor in pixels")
                .readOnly()
                .commit()
        ,
        BOOL_ELEMENT(expected).key("cameraAcquiring")
                .displayedName("Camera Acquiring")
                .description("Returns whether or not an acquisition is currently running")
                .readOnly()
                .commit()
        ,
        STRING_ELEMENT(expected).key("cameraModel")
                .displayedName("Camera Model")
                .description("Returns the camera model")
                .readOnly()
                .commit()
        ,
        )

    ##############################################
    #   Implementation of State Machine methods  #
    ##############################################

    def errorStateOnEntry(self):
        self.log.INFO("ErrorState OnEntry")
        
    def errorStateOnExit(self):
        self.log.INFO("ErrorState OnExit")
    
    def initializationStateOnEntry(self):
        self.log.INFO("InitializationState OnEntry")
        
        # Camera model
        self.set("cameraModel", "simCam")
        
        imageType = self.get("imageType")
        
        try:
            if imageType == "2d_Gaussian":
                # 2d Gaussian, no rotation
                rvx = scipy.stats.norm(300, 50)
                x = rvx.pdf(numpy.arange(800)) # 1d gaussian
                rvy = scipy.stats.norm(200, 75)
                y = rvy.pdf(numpy.arange(600)) # 1d gaussian
                z = numpy.outer(y, x) # 2d gaussian (float64)
                data = (z/z.max()*0.5*numpy.iinfo('uint8').max).astype('uint8') # -> uint8
                self.log.INFO('2d gaussian image loaded')
            elif imageType == 'RGB_Image':
                # RGB image
                data = numpy.append(numpy.append([[255,0,0]*67500], [[0,255,0]*67500]), [[0,0,255]*67500]).reshape(450,450,3).astype('uint8')
                self.log.INFO('RGB image loaded')
            elif imageType == 'Load_from_file':
                # Try to load image file
                filename = self.get("imageFilename")
                data = numpy.load(filename)
                self.log.INFO('Image loaded from file ' + filename)
            else:
                # Default image, grayscale, vertical gradient
                a = numpy.arange(500, dtype=numpy.uint16)
                b = numpy.array([a]*500)
                data = numpy.rot90(b)
                self.log.INFO('Default image (grayscale) loaded')
        except Exception as e:
            # Default image, grayscale, vertical gradient
            a = numpy.arange(500, dtype=numpy.uint16)
            b = numpy.array([a]*500)
            data = numpy.rot90(b)
            self.log.WARN(str(e))
            self.log.INFO('Default image (grayscale) loaded')
        
        self.image = data
        
        # Sensor geometry
        self.set("sensorHeight", data.shape[0])
        self.set("sensorWidth", data.shape[1])

        # Starts polling thread now
        self.pollThread.start()

    def acquisitionStateOnEntry(self):
        self.log.INFO("AcquisitionState OnEntry")
        
        # Prepares and starts acquisition thread
        self.doAcquire = True
        self.triggerReceived = False
        self.acqThread = threading.Thread(target = self.acquireImages)
        self.acqThread.start()
        
        self.set("cameraAcquiring", True);

    def acquisitionStateOnExit(self):
        self.log.INFO("AcquisitionState OnExit")
        
        # Stops acquisition thread
        self.doAcquire = False
        self.acqThread.join(10.)
        
        self.set("cameraAcquiring", False);

    def readyStateOnEntry(self):
        self.log.INFO("ReadyState OnEntry")
        
    def readyStateOnExit(self):
        self.log.INFO("ReadyState OnExit")
    
    def resetAction(self):
        self.log.INFO("resetAction")
    
    def acquireAction(self):
        self.log.INFO("acquireAction")
    
    def stopAction(self):
        self.log.INFO("stopAction")
    
    def triggerAction(self):
        self.log.INFO("triggerAction")

        self.triggerReceived = True

    def alwaysRun(self):
        self.log.INFO("alwaysRun")
        
        while self.doAlwaysRun:
            
            # Here more periodical checks can be added
            
            # Sleeps for a fraction of second
            time.sleep(0.2)
   
    def pollHardware(self):
        pollInterval = self.get("pollingInterval")
        mesg = "pollHardware %s" % pollInterval
        self.log.INFO(mesg)
        
        while self.doPolling:
            temperature = 25.4 + random.random()/10.
            self.set("sensorTemperature", temperature);

            # Sleeps for a while
            t = self.get("pollingInterval")
            time.sleep(t)

    def acquireImages(self):
        self.log.INFO("acquireImages")
        
        swTrigger  = False
        if self.get("triggerMode") == "Software":
            swTrigger  = True

        fixedMode  = False
        frameCount = 0
        if self.get("cycleMode") == "Fixed":
            fixedMode  = True
            frameCount = self.get("frameCount")
        
        # Image type
        imageType = self.get("imageType")
        
        # Get original image and multiply it by pixelGain
        pixelGain = self.get("pixelGain")
        image = self.image # Copy original image
        image *= pixelGain # Apply pixel gain to copy
        
        i = 0
        while self.doAcquire:
            
            if self.isMainThreadAlive()==False:
                # MainThread is not alive... break
                self.log.INFO("leaving acquireImages")
                break

            if swTrigger:
                # Running in SW trigger mode
                if self.triggerReceived:
                    # SW trigger received -> Reset flag
                    self.triggerReceived = False
                else:
                    # No SW trigger yet... wait 10 ms
                    time.sleep(0.010)
                    continue
            
            expTime = self.get("exposureTime")
            time.sleep(expTime) # Sleep for the appropriate time
            
            if imageType == '2d_Gaussian':
                # Add some random noise
                image2 = image + numpy.random.uniform(high=20, size=image.shape).astype('uint8')
            else:
                # Roll image by 10 lines
                w = 10*image.shape[0]
                image = numpy.roll(image, w)
                image2 = image
            
            if len(image2.shape)==2:
                # Grayscale
                rawImageData = RawImageData(image2, EncodingType.GRAY)
            elif len(image2.shape)==3 and image2.shape[2]==3:
                # RGB
                rawImageData = RawImageData(image2, EncodingType.RGB)
            elif len(image2.shape)==3 and image2.shape[2]==4:
                # RGBA
                rawImageData = RawImageData(image2, EncodingType.RGBA)
            else:
                self.log.ERROR("Image has odd shape: " + str(array.shape) + " -> skip it!")
                # Unknown
                continue
            
            try:
                # Set image element
                self.set("image", rawImageData)
            except Exception as e:
                self.log.ERROR("Could not set image")
                self.log.DEBUG(str(e))
            
            try:
                self.output.write(rawImageData) # Send the image to device output channel
                self.output.update() # Flush the output channel
            except Exception as e:
                self.log.ERROR("Could not write image to output channel")
                self.log.DEBUG(str(e))
            
            # Counter
            i += 1
            if fixedMode and i>=frameCount:
                # The desired number of frames have been taken
                self.execute("stop") # trigger transition to Ready state
                break # exit acquisition loop

    def isMainThreadAlive(self):
        # Checks whether MainThread is still alive
        
        threadlist=threading.enumerate()
        for thread in threadlist:
            if thread.name=="MainThread":
                if thread.isAlive():
                    # MainThread is alive
                    return True
                else:
                    # MainThread is not alive
                    return False

        # MainThread not found -> not alive
        return False

    def preDestruction(self):
        # Tells threads to exit
        self.doAlwaysRun = False
        self.doPolling   = False
        self.doAcquire = False
        
        # Joins the "always run" Thread
        if (self.alwaysRunThread!=None and self.alwaysRunThread.isAlive()):
            self.alwaysRunThread.join(10.)
        
        # Joins the polling Thread
        if (self.pollThread!=None and self.pollThread.isAlive()):
            self.pollThread.join(10.)
        
        # Joins the acquisition Thread
        if (self.acqThread!=None and self.acqThread.isAlive()):
            self.acqThread.join(10.)


# This entry used by device server
if __name__ == "__main__":
    launchPythonDevice()
