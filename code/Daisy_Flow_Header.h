#include "opencv2/opencv.hpp"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <set>
using std::set;
using std::max;
using std::min;

#include "Common_Datastructure_Header.h"
#include "Program_Control_Header.h"

#include "colorcode.h"
#include "FlowInputOutput.h"

#include "CF_Filter_Header.h"
#include "GFilter_Header.h"
#include "CLMF_Header.h"
#include "BoxFilter_Header.h"

#include "DenseFeatures_Header.h"

#include "Superpixels_Header.h"

#define EPS 0.01
#define DOUBLE_MAX 1e10
#define MY_PI 3.1415926535897932384626433832795
#define MY_PI_INV 0.31830988618379067153776752674503

class SuperPatchmatch
{
public:
	SuperPatchmatch();
	~SuperPatchmatch();

	// fullsize left and right image
	cv::Mat_<cv::Vec3f> imLeft, imRight;
	cv::Mat_<cv::Vec3b> imLeftOrigin, imRightOrigin;
	int widthLeft, heightLeft;
	int widthRight, heightRight;

	// upsample scale
	int upPrecision;
	cv::Mat_<cv::Vec3f> imLeftUp, imRightUp;

	float minHorPosLeft, maxHorPosLeft, horRangeLeft;
	float minVerPosLeft, maxVerPosLeft, verRangeLeft;

	float minHorPosRight, maxHorPosRight, horRangeRight;
	float minVerPosRight, maxVerPosRight, verRangeRight;

public:
	// sub-image range and super-pixel range
	// {0: left, 1: up, 2: right, 3: down}
	cv::Mat_<cv::Vec4i> subRangeLeft, subRangeRight;
	cv::Mat_<cv::Vec4i> spRangeLeft, spRangeRight;
	// number of super-pixels
	int numOfLabelsLeft, numOfLabelsRight;

	// the label of each pixel
	cv::Mat_<int> segLabelsLeft, segLabelsRight;
	// use for superpixel based graph traverse
	cv::Mat_<cv::Vec2i> repPixelsLeft, repPixelsRight;
	GraphStructure spGraphLeft, spGraphRight;

	cv::Mat_<float> bestLeftCost;
	cv::Mat_<float> bestRightCost;

#if SUPERPIXELS_VISITED_LABELLIST
	vector<bool **> spLabelHashLeft;
	//vector<int> spLabelVisitedNumberLeft;
	vector<bool **> spLabelHashRight;
	//vector<int> spLabelVisitedNumberRight;
#endif
private:
	inline bool VerifyLabelInHashLeft(int spLabel, float horLabel, float verLabel);
	inline bool VerifyLabelInHashRight(int spLabel, float horLabel, float verLabel);

public:
	int iterationTimes;
public:
	int processingFrameId;

public:
	void WriteOutImageResult(const cv::Mat &img, const char *resName, int defaultId = -1);
public:
	// error test function
	cv::Mat_<cv::Vec3b> currentFlowColorLeft;
	cv::Mat_<cv::Vec3b> currentFlowColorRight;
	cv::Mat_<cv::Vec2f> gtFlow;
	double TestFlowEndPointError(const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow, cv::Mat_<uchar> &errorImg);
	double TestFlowAngularError(const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow, cv::Mat_<uchar> &errorImg);
	float refMaxMotion;
	void ShowMotionFieldInColorCoded(const cv::Mat_<cv::Vec2f> &flowVec, const char *winName, cv::Mat_<cv::Vec3b> &flowColor, float maxMotion = -1.0);
	void ReadFlowFile(const char *flowFile, cv::Mat_<cv::Vec2f> &flowVec, int height, int width);
	void WriteFlowFile(const char *flowFile, const cv::Mat_<cv::Vec2f> &flowVec, int height, int width);
	bool hasGtFlow;

	// [buffered data], i.e. some offline processing data structure
public:
	// crossmap of left and right images
	int crossArmLength, crossColorTau;
	cv::Mat_<cv::Vec4b> crossMapLeft, crossMapRight;
	// sub-image buffer and sub-image cross map buffer
	vector<cv::Mat_<cv::Vec3f>> subImageLeft;
	vector<cv::Mat_<cv::Vec4b>> subCrossMapLeft;

	// new added for processing backward flow
	vector<cv::Mat_<cv::Vec3f>> subImageRight;
	vector<cv::Mat_<cv::Vec4b>> subCrossMapRight;

private:
	void ModifyCrossMapArmlengthToFitSubImage(const cv::Mat_<cv::Vec4b> &crMapIn, int maxArmLength, cv::Mat_<cv::Vec4b> &crMapOut);
	void InitiateBufferData();

private:
	inline void CalculateStatisticsAfterIteration();

	// [added] - 2012-10-12 to random assign pixels
public:
	vector<vector<cv::Vec2i>> superpixelsListLeft;
	vector<vector<cv::Vec2i>> superpixelsListRight;
	void GetSuperpixelsListFromSegment(const cv::Mat_<int> &segLabels, int numOfLabels, vector<vector<cv::Vec2i>> &spPixelsList);
	void RandomAssignRepresentativePixel(const vector<vector<cv::Vec2i>> &spPixelsList, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel);

	// [added] - 2012-10-29 
public:
	void BuildSuperpixelsPropagationGraph(const cv::Mat_<int> &refSegLabel, int numOfLabels, const cv::Mat_<cv::Vec3f> &refImg, GraphStructure &spGraph );
#if USE_GF_TO_FILTER_COST
	vector<GFilter> subGFLeft, subGFRight;
#endif

#if USE_CLMF1_TO_AGGREGATE_COST
	vector<CLMFilter> subCLMFLeft, subCLMFRight;
#endif

public:
//	void AssociateLeftRightImageToInitiateFlow(int sampleNum);
	void AssociateLeftImageItselfToEstablishNonlocalPropagation(int sampleNum, int topK);
	void AssociateRightImageItselfToEstablishNonlocalPropagation(int sampleNum, int topK);

public:
	void ClearUpMemory();

	//[added] - 2012-11-07
	double CalculateSumOfSubImages(vector<cv::Mat_<cv::Vec3f>> &subImage);
	double updatedLabelNumberLeft;
	double updatedLabelNumberRight;

	// [added] - 2013-03-06, for sift flow
	cv::Mat_<uchar> imgGrayLeft, imgGrayRight;

#if USE_ENHANCED_DAISY_FLOW_FEATURES
	// [added] - 2013-03-20
	enum
	{
		MAX_SCALE_LEVEL_LEFT = 1,
		MAX_SCALE_LEVEL_RIGHT = 5, //5
		MAX_ORIENTATION_LEVEL = 7, // 7 selectable value: 1, 3, 5, 7, 13		
	};
	static const int BASE_DESCRIPTOR_SIZE = 16;

	float scaleSigmaLeft[MAX_SCALE_LEVEL_LEFT]; // create the blurred image, on that sigma level
	int scaleGaussSizeLeft[MAX_SCALE_LEVEL_LEFT];
	float scaleSCoefLeft[MAX_SCALE_LEVEL_LEFT]; // corresponding scale level
	int scaleDescRadiusLeft[MAX_SCALE_LEVEL_LEFT]; // corresponding descriptor radius

	float scaleSigmaRight[MAX_SCALE_LEVEL_RIGHT]; // create the blurred image, on that sigma level
	int scaleGaussSizeRight[MAX_SCALE_LEVEL_RIGHT];
	float scaleSCoefRight[MAX_SCALE_LEVEL_RIGHT]; // corresponding scale level
	int scaleDescRadiusRight[MAX_SCALE_LEVEL_RIGHT]; // corresponding descriptor radius

	float oriAngle[MAX_ORIENTATION_LEVEL]; // angle of each orientation
	float oriCosTheta[MAX_ORIENTATION_LEVEL];
	float oriSinTheta[MAX_ORIENTATION_LEVEL]; // the cos/sin value of corresponding angle

	void InitializeDaisyFeatures();
	daisy *descLeft[MAX_SCALE_LEVEL_LEFT];
	daisy *descRight[MAX_SCALE_LEVEL_RIGHT];

	void RunDaisyFilterFlow(cv::Mat_<cv::Vec2f> &flowResult);
	cv::Mat_<cv::Vec4f> bestLeftDaisyFlow, bestRightDaisyFlow;
	void ImproveDaisyFlowLabelListLeft(int py, int px, vector<cv::Vec4f> &flowList);
	void ImproveDaisyFlowLabelListRight(int py, int px, vector<cv::Vec4f> &flowList);

	vector<cv::Mat_<float>> subDaisyLeft, subDaisyRight;

	// for visualization
	void CopyFirstTwoChannelsToFlow(const cv::Mat_<cv::Vec4f> &flIn, cv::Mat_<cv::Vec2f> &flOut);
	void CopySelectedChannelToFloat(const cv::Mat_<cv::Vec4f> &floatIn, int selCh, cv::Mat_<float> &floatOut);

#endif

	void CreateMaskedImage(const cv::Mat_<cv::Vec3b> &imgIn, const cv::Mat_<uchar> &mask, cv::Mat_<cv::Vec3b> &imgOut);

	vector<vector<cv::Vec4f>> spFlowVisitedLeft;
	vector<int> spFlowVisitedNumberLeft;

	vector<vector<cv::Vec4f>> spFlowVisitedRight;
	vector<int> spFlowVisitedNumberRight;

	inline bool VerifyFlowInVisitedListLeft(int spLabel, const cv::Vec4f &fl);
	inline bool VerifyFlowInVisitedListRight(int spLabel, const cv::Vec4f &fl);

	// [added] - 2013-04-09
	void WarpingForwardAffine(const cv::Mat_<cv::Vec4f> &flowMapping, const cv::Mat_<cv::Vec3b> &srcImg, int dstWidth, int dstHeight, cv::Mat_<cv::Vec3b> &dstImg, const float *scaleSCoef);
	void WarpingReverseAffine(const cv::Mat_<cv::Vec4f> &flowMapping, const cv::Mat_<cv::Vec3b> &dstImg, cv::Mat_<cv::Vec3b> &srcImg, const float *scaleSCoef);

	void ConvertAffineParametersToFlow(const cv::Mat_<cv::Vec4f> &affFlow, cv::Mat_<cv::Vec2f> &motionFlow, const float *scaleSCoef);

	void CrossCheckToCreateConfidenceMask(const cv::Mat_<cv::Vec4f> &fwFlow, const cv::Mat_<cv::Vec4f> &bwFlow, float radiusThresh, cv::Mat_<uchar> &srcConfMask, cv::Mat_<uchar> &dstConfMask);


	// [added] - 2013-04-12
	bool useMaskTransfer;
	cv::Mat_<cv::Vec3b> loadedGtMask;
	void TransferMaskUsingFlow(const cv::Mat_<cv::Vec3b> &maskIn, const cv::Mat_<cv::Vec2f> &motionFlow, cv::Mat_<cv::Vec3b> &maskOut);

	// [added] - 2013-09-29, calculate scaled ground truth flow
	float scaleLeftFactor, scaleRightFactor;
	void CalculateScaledGroundTruthFlow(const cv::Mat_<cv::Vec2f> &flowIn, cv::Mat_<cv::Vec2f> &flowOut);

	void PostRefineUsingBilateralFilterAsGuidance( const cv::Mat_<cv::Vec2f> &flowVec, const cv::Mat_<cv::Vec3b> &weightColorImg, int radius, float bfSigmaSpatial, float bfSigmaColor, cv::Mat_<cv::Vec2f> &refinedFlow );


	// [added - 2013-10-25], calculate error
	double CalcFlowStandardDeviation(const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow);
	double CalcDiceCoefficient(const cv::Mat_<cv::Vec3b> &maskIn, const cv::Mat_<cv::Vec3b> &maskRef);

	// ====================================================================
	// [added] - 08-2014
	int SetDefaultParameters();
	int CreateAndOrganizeSuperpixels();
	// =====================================================================
	// common parameters list
	// filter kernel related
	int g_filterKernelSize;
	int g_filterKernelBoundarySize;
	int g_filterKernelColorTau;
	int g_filterKernelEpsl;
	// superpixel related
	int g_spMethod;
	int g_spNumber;
	int g_spSize;
	int g_spSizeOrNumber;
	// process flow
	int g_iterTimes;
	int g_flowPrecision;

	// process left to right frame flow
	bool DO_LEFT;
	// process right to left frame flow
	bool DO_RIGHT;
};











