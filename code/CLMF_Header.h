#ifndef __CLMF1_FILTER_HEADER
#define __CLMF1_FILTER_HEADER

#include "opencv2/opencv.hpp"
#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

class CLMFilter
{
public:
	cv::Mat_<float> varIBB, varIBG, varIBR, varIGG, varIGR, varIRR;
	cv::Mat_<float> mIBB, mIBG, mIBR, mIGG, mIGR, mIRR;
	cv::Mat_<float> mIR, mIG, mIB;
	cv::Mat_<float> mIRp, mIGp, mIBp;
	cv::Mat_<float> covIRp, covIGp, covIBp;
	cv::Mat_<float> mP;
	cv::Mat_<float> normWin;
	vector<vector<cv::Matx33f>> invIU; 

	cv::Mat_<float> aPR, aPG, aPB;
	cv::Mat_<float> bP;

	cv::Mat_<float> maPR, maPG, maPB;
	cv::Mat_<float> mbP;

	cv::Mat_<float> iR, iG, iB;

	cv::Mat_<cv::Vec3b> guidedI;
	cv::Mat_<float> costP;

	cv::Mat_<cv::Vec4b> crossMap;

	void GetCrossUsingSlidingWindow(const cv::Mat_<cv::Vec3b> &img, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau);
	inline void TheCrossBoxFilter(const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, const cv::Mat_<cv::Vec4b> &crMap);
	void InitiateGuidance(const cv::Mat_<cv::Vec3b> &gImg, const cv::Mat_<cv::Vec4b> &crMap, float epsl);
	void DoCLMF1Filter(const cv::Mat_<float> &costIn, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat_<float> &filteredCost);
	void DoCLMF0Filter(const cv::Mat_<float> &costIn, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat_<float> &filteredCost);
};



#endif