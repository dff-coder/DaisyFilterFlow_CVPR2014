#ifndef __OPENCV_TICKCOUNT_HEADER
#define __OPENCV_TICKCOUNT_HEADER

/// ENV:
/// OpenCV 2.0+?
/// REM:
/// Using OpenCV getTickCount to measure time interval

#include "opencv2/opencv.hpp"

class CalcTime
{
private:
	double st, end;
	
public:
	CalcTime() : st(0.0), end(0.0){};
	void Reset()
	{
		st = end = 0.0;
	}
	void Start()
	{
		st = (double)cv::getTickCount();
	}

	double End()
	{
		end = (double)cv::getTickCount();
		return (end-st)*1000.0/cv::getTickFrequency();
	}

	double End(const char *str)
	{
		double tmp = End();
		printf("%s Costs %.6f ms\n", str, tmp);
		return tmp;
	}
};

#endif