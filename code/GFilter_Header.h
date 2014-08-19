#ifndef __GUIDED_FILTER_HEADER
#define __GUIDED_FILTER_HEADER

#include "opencv2/opencv.hpp"
#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

class GFilter
{
public:
	cv::Mat_<float> mIR, mIG, mIB;
	cv::Mat_<float> normWin;
	vector<vector<cv::Matx33f>> invIU; 

	cv::Mat_<float> iR, iG, iB;

	void TheBoxFilterArrayForm( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius );
	void InitiateGuidance(const cv::Mat_<cv::Vec3b> &gImg, int radius, float epsl);
	void InitiateGuidance(const cv::Mat_<cv::Vec3f> &gImg, int radius, float epsl);
	void DoGuidedFilter(const cv::Mat_<float> &costIn, int radius, float epsl, cv::Mat_<float> &filteredCost);
public:
	void ClearUp();
	~GFilter();

public:
	// [added] - 2012-11-09, try optimize a bit
	void NewDoGuidedFilter( const cv::Mat_<float> &costIn, int radius, float epsl, cv::Mat_<float> &filteredCost );
	inline void TheBoxFilter(const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius);
};



#endif