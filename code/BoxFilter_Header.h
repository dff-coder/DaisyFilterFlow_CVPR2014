#ifndef __BOX_FILTER_HEADER
#define __BOX_FILTER_HEADER

#include "opencv2/opencv.hpp"

class CostBoxFilter
{
public:
	cv::Mat_<float> normWin;
	void DoCostBoxFilter( const cv::Mat_<float> &costIn, int radius, cv::Mat_<float> &filteredCost );
	inline void TheBoxFilter(const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius);
	inline void TheBoxFilterArrayForm( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius );
};



#endif