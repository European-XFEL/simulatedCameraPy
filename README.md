# SimulatedCameraPy for the Karabo framework

## Overview

This package contains a single Karabo device called SimulatedCameraPy
that simulates a camera image that shows a beam profile.

The device class inherits from the ImageSourcePy class from the 'imageSourcePy' package.
Therefore its images are available via an output channel under the schema key
'output.schema.data.image'.

## Installation

Within an activated Karabo environment, use

karabo install simulatedCameraPy <tag>

It might be needed to specify the repository via the -g option of the karabo command.

## Dependencies

This package depends on the 'imageSourcePy' Karabo package.


## Contact

For questions, please contact opensource@xfel.eu.


## License and Contributing

This software is released by the European XFEL GmbH as is and without any
warranty under the GPLv3 license.
If you have questions on contributing to the project, please get in touch at
opensource@xfel.eu.
