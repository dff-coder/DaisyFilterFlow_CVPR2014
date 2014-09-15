DaisyFilterFlow_CVPR2014
========================

This repo contains the main code of daisy filter flow which is published in CVPR 2014. ~~Please check it later after Aug 2014.~~

- v.1 - Aug 2014
  - A guided filter enabled version is provided. 
  - *note:* our run-time measurement and results reported in paper was using CLMF filters. The CLMF-0 gives similar quality to GF, but runs 2x faster. It would be added later.

Paper Website: [https://sites.google.com/site/daisyfilterflowcvpr2014/](https://sites.google.com/site/daisyfilterflowcvpr2014/)

## Usage
- The essential code is under the `code` folder
- For windows user, see `sample_executable` folder for a sample usage. (Runtime DLLs are included)
- Please check the `CMakeLists.txt` file for the programs structure. You can directly generate compatible project using [CMake](http://www.cmake.org/). (Tested only in Windows platform and using Visual Studio for now; but the code should be able to run in Linux or Mac with slight modification). 

## Parameters Tuning Guideline
- **First of all**, please refer to both [Daisy Filter Flow](https://sites.google.com/site/daisyfilterflowcvpr2014/) and [Patchmatch Filter](https://sites.google.com/site/daisyfilterflowcvpr2014/) papers for the meaning and strategy of each used parameters. 
- Set superpixels number or size accordingly to image size. Current setting (superpixel size = 300) is good for typical images of 320*240. It's not sensitive but could be important for the converge speed and performance.
- The parameters related to filtering method should be adjusted for the data, e.g. kernel size, epsl of GF, color tau for CLMF0, CLMF1 (support would be added in the future update), etc.
- Raw cost function should be noticed: L1 or L2 distance, truncated value, truncated way.
- The other key parameters has relatively little effect but still should be noticed:
	- Number of associated affinity neighbour superpixels
	- The sampling decreased ratio alpha for scale, orientation, translation 
	- The sampling range in scale, orientation
	- About daisy descriptors computing: convolved histogram sampled layers, the way to handling boundary of image, descriptor size, descriptor sampling layers, etc.

## Dependency
- OpenCV > 2.3+, [http://opencv.org/](http://opencv.org/)  
Mostly the code only use opencv's cv::Mat structure to handling input and output image data.
- SLIC superpixel, (included), [http://ivrg.epfl.ch/research/superpixels](http://ivrg.epfl.ch/research/superpixels)
- daisy descriptor, (modified and included), [http://cvlab.epfl.ch/software/daisy](http://cvlab.epfl.ch/software/daisy)
- optical flow color coding code, (wrapped and included) [http://vision.middlebury.edu/flow/submit/](http://vision.middlebury.edu/flow/submit/)

## License
For research and education purpose only. For feedback and bug issues, please send email to yhs@cs.unc.edu. 
