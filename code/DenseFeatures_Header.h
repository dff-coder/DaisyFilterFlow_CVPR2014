#ifndef __DENSE_FEATURES_HEADER
#define __DENSE_FEATURES_HEADER

#define M_PI 3.1415926535897932384626433832795

class daisy;

int CreateDenseDaisy(const cv::Mat &imgIn, float scale, float angle, cv::Mat &descrpOut);

// [added] - 2013-03-20, to pre-buffering daisy flow
daisy *InitializeOneDaisyDesc(const cv::Mat &imgIn, float scale, int descLayers = 2);
int ExtractSubImageDaisyDescriptors(daisy *desc, cv::Mat_<float> &descOut, float py, float px, float step, int ori, int h, int w, float limH, float limW);
int ExtractAndComputeSubImageDaisyDescriptorsCost(daisy *desc, cv::Mat_<float> &costOut, cv::Mat_<float> &descRef, float py, float px, float step, int ori, int h, int w, cv::Vec4f &tmpFl);

// [added] - 2013-04-09, correct the affine transformation
int NewlyExtractAndComputeSubImageDaisyDescriptorsCost( daisy *desc, cv::Mat_<float> &costOut, const cv::Mat_<float> &descRef, float oy, float ox, int h, int w, float dy, float dx, float step, int ori);

#endif