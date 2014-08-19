cmake_minimum_required(VERSION 2.7)	
project(Daisy_Filter_Flow)

# set the proper path of header and library files of OpenCV
set(OPENCV_INCLUDE $ENV{OPENCV_X64_INCLUDE})
set(OPENCV_LIB_PATH $ENV{OPENCV_X64_LIB})

include_directories(
	${OPENCV_INCLUDE}
	./code/daisy_descriptor/include
)

link_directories(
	${OPENCV_LIB_PATH}
)

file(GLOB DAISY_DESCRIPTOR 
code/daisy_descriptor/src/*)

file(GLOB FILTER_FILE 
code/CLMF* 
code/GFilter* 
code/BoxFilter* 
code/CF*)

file(GLOB SUPERPIXEL_FILE 
code/SLIC* 
code/Superpixels*)

file(GLOB FLOW_MANIPULATION_FILE 
code/colorcode* 
code/FlowInputOutput*)

file(GLOB COMMON_FILE 
code/Common_DataStructure_*
code/Image_Video_Capture*
code/OpenCV_Tickcount*
code/Scan_File*)


file(GLOB DAISY_FLOW_FILE code/Daisy_Flow* code/DenseFea* code/General_* code/Program_Control*)

if(MSVC)
   add_definitions(/arch:SSE2)
endif(MSVC)


add_executable(Daisy_Filter_Flow
	./sample_executable/example_DaisyFilterFlow_main.cpp
	${DAISY_DESCRIPTOR}
	${FILTER_FILE}
	${SUPERPIXEL_FILE}
	${FLOW_MANIPULATION_FILE}
	${COMMON_FILE}
	${DAISY_FLOW_FILE}
)

target_link_libraries(Daisy_Filter_Flow 
	debug opencv_highgui248d.lib optimized opencv_highgui248.lib
	debug opencv_core248d.lib	optimized opencv_core248.lib
	debug opencv_imgproc248d.lib optimized opencv_imgproc248.lib
#	debug opencv_features2d248d.lib optimized opencv_features2d248.lib
#	debug opencv_nonfree248d.lib optimized opencv_nonfree248.lib
)

source_group("Daisy_Descriptor" FILES ${DAISY_DESCRIPTOR})
source_group("Filter_Modules" FILES ${FILTER_FILE})
source_group("Superpixel_Modules" FILES ${SUPERPIXEL_FILE})
source_group("Flow_Manipulation_Modules" FILES ${FLOW_MANIPULATION_FILE})
source_group("Common_Files" FILES ${COMMON_FILE})