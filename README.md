# CUDA At Scale For Enterprise Final Project V2
The second project I did for the module "CUDA at Scale for Enterprise" from Coursera.

## Project Description

This project load an image sequence, a set of images, or a video stream and process the magnitude of its images after filtering them using horizontal and vertical Sobel filter. The processing is made in batches. First, each image is filtered independently by the Sobel filters (using NPPI). The result of each filters is placed into a batch, (one batch per direction). The batches are full the magnitude of all the images of the batch is computed at once. Finaly the images are are written either in a video file or in a folder depending on the arguments provided to the function.
It create a command line funtion named ```cudaAtScaleV2.exe```.

### Arguments for ```cudaAtScaleV2.exe```:
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

```bin/```
This folder will only exist once the project is compiled, and will contains: ```cudaAtScaleV2.exe``` and ```unit_test.exe``` if the unit test was build.

```data/```
This folder contains few images, which originally comes from the data folder of the sample folder of the legacy modules of OpenCV (i.e. from [here](https://github.com/opencv/opencv/tree/4.x/samples/data) )

```src/```
All the files that this project depends on are here.

```test/```
All the files that related to the build of the unit test.

```INSTALL```
This files gives some advise regarding how to install the dependencies of this project (i.e. CUDA, NPPI, OpenCV).

```CMAkeLists.txt```
Configuration file.

```run.sh```
Run this file in order to compile the project.


## Notes And Observations

Other than CUDA and NPPI this project requires OpenCV.
Unless you are comfortable with working with custom OpenCV versions it is advanced when available to use OpenCV's version of the package manager.
Ubuntu's apt has libopencv-dev which fits the requirement of this project.

For one who want to try to use an image sequence as input, please keep in mind that if the formating of your files is interupted (e.g. left01.jpg,..., left.09.jpg, left11.jpg,...left14.jpg), only the files before the interuption will be seen.
