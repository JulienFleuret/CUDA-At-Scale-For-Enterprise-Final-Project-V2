# CUDA At Scale For Enterprise Final Project V2
The second project I did for the module "CUDA at Scale for Enterprise" from Coursera.

## Project Description

This project loads an image sequence, a set of images, or a video stream and processes the magnitude of its images after filtering them using horizontal and vertical Sobel filters. The processing is made in batches. First, each image is filtered independently by the Sobel filters (using NPPI). The result of each filter is placed into a batch, (one batch per direction). Once all the images of the batch have been filtered the magnitude of the batch is computed at once. Finally the images are written either in a video file or in a folder depending on the arguments provided to the function.
It creates a command line function named `cudaAtScaleV2.exe`.

### Arguments for `cudaAtScaleV2.exe`:
- `-h`, `-help`, `-usage`, or `-?`: Print help.
- `-input`,`-i`, or `-if` : specify the input which can be:
  - a video file (e.g. smth.avi, anthg.mp4).
  - a video stream (eg. protocol://host:port/script_name?script_params|auth).
  - an image sequence (eg. img_%02d.jpg)
- `-output`,`-o`, or `-of` : specify the input which can be:
  - a video file (e.g. smth.avi, anthg.mp4).
  - an image sequence (eg. img_%02d.jpg)
- `-batch_size`, or `-bs`: (optional, default: -1) batch size to use for the computations.

## Code Organization

`bin/`
This folder will only exist once the project is compiled, and will contain: `cudaAtScaleV2.exe` and `unit_test.exe` if the unit test was built.

`data/`
This folder contains a few images, which originally came from the data folder of the sample folder of the legacy modules of OpenCV (i.e. from [here](https://github.com/opencv/opencv/tree/4.x/samples/data) )

`src/`
All the files that this project depends on are here.

`test/`
All the files related to the build of the unit test.

`INSTALL`
This file gives some advice regarding how to install the dependencies of this project (i.e. CUDA, NPPI, OpenCV).

`CMAkeLists.txt`
Configuration file.

`run.sh`
Run this file in order to compile the project.


## Notes And Observations

Other than CUDA and NPPI this project requires OpenCV.
Unless you are comfortable with working with custom OpenCV versions it is advised when available to use OpenCV's version of the package manager.
Ubuntu's `apt` has `libopencv-dev` which fits the requirement of this project.

For one who wants to try to use an image sequence as input, please keep in mind that if the formatting of your files is interrupted (e.g. `left01.jpg`,..., `left.09.jpg`, `left11.jpg`,...,`left14.jpg`), only the files before the interruption will be seen.

For someone working in ubuntu the package `opencv-doc` will install the OpenCV's sample folder at this location: `/usr/share/doc/opencv-doc/examples/data/`.
You can try for instance `./bin/cudaAtScaleV2.exe --input=/usr/share/doc/opencv-doc/examples/data/left%02d.jpg --output=wherever/you/whish/result%02.jpg` 
