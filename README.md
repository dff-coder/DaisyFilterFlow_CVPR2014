DaisyFilterFlow_CVPR2014
========================

This repo contains the main code of daisy filter flow which is published in CVPR 2014. ~~Please check it later after Aug 2014.~~

v.1 - Aug 2014

Paper Website: [https://sites.google.com/site/daisyfilterflowcvpr2014/](https://sites.google.com/site/daisyfilterflowcvpr2014/)

## Usage
- The essential code is under the `code` folder
- For windows user, see `sample_executable` folder for a sample usage. (Runtime DLLs are included)
- Please check the `CMakeLists.txt` file for the programs structure. You can directly generate compatible project using [CMake](http://www.cmake.org/). (Tested only in Windows platform and using Visual Studio for now; but the code should be able to run in Linux or Mac with slight modification). 

## Dependency
- OpenCV > 2.3+, [http://opencv.org/](http://opencv.org/)  
Mostly the code only use opencv's cv::Mat structure to handling input and output image data.
- SLIC superpixel, (included), [http://ivrg.epfl.ch/research/superpixels](http://ivrg.epfl.ch/research/superpixels)
- daisy descriptor, (modified and included), [http://cvlab.epfl.ch/software/daisy](http://cvlab.epfl.ch/software/daisy)
- optical flow color coding code, (wrapped and included) [http://vision.middlebury.edu/flow/submit/](http://vision.middlebury.edu/flow/submit/)

## License
For research and education purpose only. For feedback and bug issues, please send email to yhs@cs.unc.edu. 
