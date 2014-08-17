#ifndef __IMAGE_VIDEO_CAPTURE_HEADER
#define __IMAGE_VIDEO_CAPTURE_HEADER

#include "opencv2\opencv.hpp"

#include <string>

class ImageVideoCapture
{
public:
	enum SourceType
	{
		LOAD_IMAGE = 0,
		LOAD_VIDEO = 1,
		LOAD_WEBCAM = 2
	};

	ImageVideoCapture(SourceType sourceType);
	~ImageVideoCapture();

	void ChangeSourceType(SourceType newSourceType);

	// for image, video file
	void OpenTheSource(std::string &filename);
	// for webcam id
	// later will add videoInput support to choose the camera ID
	void OpenTheSource(int camId);

	// read video or webcam frame
	int ReadFrame();
	int ReadFrame(int frameId);
	void ResetVideoPostion();

	// these functions work for video file
	int GetNumberOfFrames();
	int GetCurrentPosition();

	// the captured image
	cv::Mat img;

private:
	SourceType st;
	std::string sourceFilename;
	int cameraID;
	cv::VideoCapture cap;
	int framePos;
};


ImageVideoCapture::ImageVideoCapture(SourceType sourceType)
{
	st = sourceType;
}

ImageVideoCapture::~ImageVideoCapture()
{
	
}

void ImageVideoCapture::ChangeSourceType(SourceType newSourceType)
{
	st = newSourceType;
}

// for image, video file
void ImageVideoCapture::OpenTheSource(std::string &filename)
{
	sourceFilename = filename;
	if (st == LOAD_IMAGE)
	{
		img = cv::imread(sourceFilename, 1);
	}
	else if (st == LOAD_VIDEO)
	{
		cap.open(sourceFilename);
	}
}

// for webcam id
// later will add videoInput support to choose the camera ID
void ImageVideoCapture::OpenTheSource(int camId)
{
	cameraID = camId;
	if (st == LOAD_WEBCAM)
	{
		cap.open(cameraID);
	}
}

// read video or webcam frame
int ImageVideoCapture::ReadFrame()
{
	if (st == LOAD_VIDEO)
	{
		return (cap.read(img));
	}
	else if (st == LOAD_WEBCAM)
	{
		return (cap.read(img));
	}
	return 1;
}

int ImageVideoCapture::ReadFrame(int frameId)
{
	if (st == LOAD_VIDEO)
	{
		cap.set(CV_CAP_PROP_POS_FRAMES, frameId);
		return (cap.read(img));
	}
	return 1;
}

void ImageVideoCapture::ResetVideoPostion()
{
	if (st == LOAD_VIDEO)
	{
		cap.open(sourceFilename);
	}
	else if (st == LOAD_WEBCAM)
	{
		cap.open(cameraID);
	}
}

// these functions work for video file
int ImageVideoCapture::GetNumberOfFrames()
{
	if (st == LOAD_VIDEO)
	{
		return cap.get(CV_CAP_PROP_FRAME_COUNT);
	}
	return 1;
}

int ImageVideoCapture::GetCurrentPosition()
{
	if (st == LOAD_VIDEO)
	{
		return cap.get(CV_CAP_PROP_POS_FRAMES);
	}
	return 0;
}




/// end of file
#endif