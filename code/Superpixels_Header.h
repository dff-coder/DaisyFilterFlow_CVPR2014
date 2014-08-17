#ifndef __SUPERPIXELS_HEADER
#define __SUPERPIXELS_HEADER

#include "opencv2\opencv.hpp"

void DrawContoursAroundSegments(const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int>	&labels, cv::Mat_<cv::Vec3b> &imgOut);

int CreateGridSegments(const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int> &segLabels, int M, int N);
int CreateSLICSegments(const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int> &segLabels, int numSp, int spSize, int createType, double compactness = 20.0);

void GetSegmentsRepresentativePixels(const cv::Mat_<int> &segLabels, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel);
void GetSegmentsRepresentativePixelsRandomAssign(const cv::Mat_<int> &segLabels, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel);


void GetSubImageRangeFromSegments(const cv::Mat_<int> &segLabels, int numOfLabels, int kerLen, cv::Mat_<cv::Vec4i> &subRange, cv::Mat_<cv::Vec4i> &spRange);
void GetPCentricSubImageRange(const cv::Mat_<cv::Vec3b> &imgIn, int spLen, int kerLen, cv::Mat_<cv::Vec4i> &subRange, cv::Mat_<cv::Vec4i> &spRange);

#endif