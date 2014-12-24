#include "Daisy_Flow_Header.h"
#include "OpenCV_Tickcount_Header.h"

SuperPatchmatch::SuperPatchmatch()
{
	srand(time(NULL));
}

SuperPatchmatch::~SuperPatchmatch()
{

}

#pragma region Evaluation_Part

inline void SuperPatchmatch::CalculateStatisticsAfterIteration()
{
#if TEST_DEBUG_INTERMEDIATE_RESULT
	// motion flow
	cv::Mat_<cv::Vec2f> motionLeft, motionRight;

	printf("NEW_Stat: Updated Label Number Left = %f, %.4f%%\n", updatedLabelNumberLeft, 100.0*updatedLabelNumberLeft/(imLeft.rows*imLeft.cols));
	printf("NEW_Stat: Updated Label Number Right = %f, %.4f%%\n", updatedLabelNumberRight, 100.0*updatedLabelNumberRight/(imRight.rows*imRight.cols));

	ConvertAffineParametersToFlow(bestLeftDaisyFlow, motionLeft, scaleSCoefRight);
	ConvertAffineParametersToFlow(bestRightDaisyFlow, motionRight, scaleSCoefLeft);
	ShowMotionFieldInColorCoded(motionLeft, WINDOW_CURRENT_FLOW_LEFT, currentFlowColorLeft);
	ShowMotionFieldInColorCoded(motionRight, WINDOW_CURRENT_FLOW_RIGHT, currentFlowColorRight);

	if (hasGtFlow)
	{
		cv::Mat_<uchar> errImg;
		printf("AEE Error: %f\n", TestFlowEndPointError(motionLeft, gtFlow, errImg));
		cv::imshow(WINDOW_ERROR_EE_IMG, errImg);
		printf("AAE Error: %f\n", TestFlowAngularError(motionLeft, gtFlow, errImg));
		cv::imshow(WINDOW_ERROR_AE_IMG, errImg);
		printf("Flow Standard Deviation: %f\n", CalcFlowStandardDeviation(motionLeft, gtFlow));
	}

	cv::Mat_<cv::Vec3b> fwFwDstImg, fwReSrcImg;
	cv::Mat_<cv::Vec3b> bwFwDstImg, bwReSrcImg;

	WarpingForwardAffine(bestLeftDaisyFlow, imLeftOrigin, widthRight, heightRight, fwFwDstImg, scaleSCoefRight);
	WarpingReverseAffine(bestLeftDaisyFlow, imRightOrigin, fwReSrcImg, scaleSCoefRight);

	cv::imshow(WINDOW_FW_FORWARD_WARPING_DST, fwFwDstImg);
	cv::imshow(WINDOW_FW_REVERSE_WARPING_SRC, fwReSrcImg);

	WriteOutImageResult(fwFwDstImg, WINDOW_FW_FORWARD_WARPING_DST);
	WriteOutImageResult(fwReSrcImg, WINDOW_FW_REVERSE_WARPING_SRC);

	WarpingForwardAffine(bestRightDaisyFlow, imRightOrigin, widthLeft, heightLeft, bwFwDstImg, scaleSCoefLeft);
	WarpingReverseAffine(bestRightDaisyFlow, imLeftOrigin, bwReSrcImg, scaleSCoefLeft);
	cv::imshow(WINDOW_BW_FORWARD_WARPING_DST, bwFwDstImg);
	cv::imshow(WINDOW_BW_REVERSE_WARPING_SRC, bwReSrcImg);

	WriteOutImageResult(bwFwDstImg, WINDOW_BW_FORWARD_WARPING_DST);
	WriteOutImageResult(bwReSrcImg, WINDOW_BW_REVERSE_WARPING_SRC);


	// [added] - 2013-03-07, for debug
	//	cv::Mat normDisLeft, normDisRight;
	//	NormalizeMaxZeroAndShow<float>(bestLeftCost, WINDOW_MATCHING_DISTANCE_LEFT, normDisLeft);
	//	NormalizeMaxZeroAndShow<float>(bestRightCost, WINDOW_MATCHING_DISTANCE_RIGHT, normDisRight);
	//	WriteOutImageResult(normDisLeft, WINDOW_MATCHING_DISTANCE_LEFT);
	//	WriteOutImageResult(normDisRight, WINDOW_MATCHING_DISTANCE_RIGHT);

#if USE_ENHANCED_DAISY_FLOW_FEATURES
	cv::Mat_<float> tmpScaleLeft, tmpScaleRight;
	cv::Mat_<uchar> normScaleLeft, normScaleRight;
	CopySelectedChannelToFloat(bestLeftDaisyFlow, 2, tmpScaleLeft);
	tmpScaleLeft = tmpScaleLeft / MAX_SCALE_LEVEL_RIGHT;
	tmpScaleLeft.convertTo(normScaleLeft, CV_8UC1, 255.0);
	WriteOutImageResult(normScaleLeft, "Norm_Scale_Left");

	CopySelectedChannelToFloat(bestRightDaisyFlow, 2, tmpScaleRight);
	tmpScaleRight = tmpScaleRight / MAX_SCALE_LEVEL_LEFT;
	tmpScaleRight.convertTo(normScaleRight, CV_8UC1, 255.0);
	WriteOutImageResult(normScaleRight, "Norm_Scale_Right");

	cv::Mat_<float> tmpOrientationLeft, tmpOrientationRight;
	cv::Mat_<uchar> normOrientationLeft, normOrientationRight;
	CopySelectedChannelToFloat(bestLeftDaisyFlow, 3, tmpOrientationLeft);
	tmpOrientationLeft = tmpOrientationLeft / MAX_ORIENTATION_LEVEL;
	tmpOrientationLeft.convertTo(normOrientationLeft, CV_8UC1, 255.0);
	WriteOutImageResult(normOrientationLeft, "Norm_Orientation_Left");

	CopySelectedChannelToFloat(bestRightDaisyFlow, 3, tmpOrientationRight);
	tmpOrientationRight = tmpOrientationRight / MAX_ORIENTATION_LEVEL;
	tmpOrientationRight.convertTo(normOrientationRight, CV_8UC1, 255.0);
	WriteOutImageResult(normOrientationRight, "Norm_Orientation_Right");
#endif

	cv::Mat_<uchar> confMaskLeft, confMaskRight, tmpMat;
	CrossCheckToCreateConfidenceMask(bestLeftDaisyFlow, bestRightDaisyFlow, 15.0, confMaskLeft, tmpMat);
	CrossCheckToCreateConfidenceMask(bestRightDaisyFlow, bestLeftDaisyFlow, 15.0, confMaskRight, tmpMat);

	cv::Mat_<cv::Vec3b> confMaskedImgLeft, confMaskedImgRight;
	CreateMaskedImage(imLeftOrigin, confMaskLeft, confMaskedImgLeft);
	WriteOutImageResult(confMaskedImgLeft, "Confidence_Masked_Left");
	CreateMaskedImage(imRightOrigin, confMaskRight, confMaskedImgRight);
	WriteOutImageResult(confMaskedImgRight, "Confidence_Masked_Right");

	CreateMaskedImage(fwReSrcImg, confMaskLeft, confMaskedImgLeft);
	WriteOutImageResult(confMaskedImgLeft, "Confidence_Masked_Reconstructed_Left");
	CreateMaskedImage(bwReSrcImg, confMaskRight, confMaskedImgRight);
	WriteOutImageResult(confMaskedImgRight, "Confidence_Masked_Reconstructed_Right");

	if (useMaskTransfer == true)
	{
		cv::Mat_<cv::Vec2f> tmpMotion;
		ConvertAffineParametersToFlow(bestLeftDaisyFlow, tmpMotion, scaleSCoefRight);
		cv::Mat_<cv::Vec3b> outMask;
		TransferMaskUsingFlow(loadedGtMask, tmpMotion, outMask);
		WriteOutImageResult(outMask, "Transfered_Mask");
	}
	fflush(stdout);
	cv::waitKey(10);
#endif
}
#pragma endregion 

void SuperPatchmatch::InitiateBufferData()
{	
	int iy;
	if (DO_LEFT)
	{	
		subImageLeft.clear();
		subImageLeft.resize(numOfLabelsLeft);

		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			int py, px;
			py = repPixelsLeft[iy][0][0];
			px = repPixelsLeft[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeLeft[py][px][2]-subRangeLeft[py][px][0]+1;
			int h = subRangeLeft[py][px][3]-subRangeLeft[py][px][1]+1;
			int x = subRangeLeft[py][px][0];
			int y = subRangeLeft[py][px][1];

			subImageLeft[iy] = imLeft(cv::Rect(x, y, w, h)).clone();
		}
	}

	if (DO_RIGHT)
	{
		subImageRight.clear();
		subImageRight.resize(numOfLabelsRight);

		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			int py, px;
			py = repPixelsRight[iy][0][0];
			px = repPixelsRight[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeRight[py][px][2]-subRangeRight[py][px][0]+1;
			int h = subRangeRight[py][px][3]-subRangeRight[py][px][1]+1;
			int x = subRangeRight[py][px][0];
			int y = subRangeRight[py][px][1];

			subImageRight[iy] = imRight(cv::Rect(x, y, w, h)).clone();
		}
	}

	// upsample image
	cv::resize(imRight, imRightUp, cv::Size(widthRight*upPrecision, heightRight*upPrecision), 0.0, 0.0, CV_INTER_CUBIC);

	// upsample and compute gradient of the left one
	cv::resize(imLeft, imLeftUp, cv::Size(widthLeft*upPrecision, heightLeft*upPrecision), 0.0, 0.0, CV_INTER_CUBIC);

#if USE_GF_TO_FILTER_COST
	int gfRadius = g_filterKernelSize;
	float gfEpsl = g_filterKernelEpsl/10000.0;

	if (DO_LEFT)
	{
		subGFLeft.clear();
		subGFLeft.resize(numOfLabelsLeft);
		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			subGFLeft[iy].InitiateGuidance(subImageLeft[iy], gfRadius, gfEpsl);
		}
	}

	if (DO_RIGHT)
	{
		subGFRight.clear();
		subGFRight.resize(numOfLabelsRight);
		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			subGFRight[iy].InitiateGuidance(subImageRight[iy], gfRadius, gfEpsl);	
		}
	}

#endif

#if USE_CLMF0_TO_AGGREGATE_COST | USE_CLMF1_TO_AGGREGATE_COST
	// initiate cross-map of image
	// various filter options
	crossColorTau = g_filterKernelColorTau;
	crossArmLength = g_filterKernelSize;

	// calculate sub-image and sub-crossmap
	cv::Mat imLeftBlur, imRightBlur;
	cv::medianBlur(imLeft, imLeftBlur, 3);
	cv::medianBlur(imRight, imRightBlur, 3);
	CFFilter cf;
	if (DO_LEFT)
	{
		cf.WrapperForSkelonBuild(imLeftBlur, crossArmLength, crossMapLeft);
		subCrossMapLeft.clear();
		subCrossMapLeft.resize(numOfLabelsLeft);

		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			int py, px;
			py = repPixelsLeft[iy][0][0];
			px = repPixelsLeft[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeLeft[py][px][2]-subRangeLeft[py][px][0]+1;
			int h = subRangeLeft[py][px][3]-subRangeLeft[py][px][1]+1;
			int x = subRangeLeft[py][px][0];
			int y = subRangeLeft[py][px][1];

			cv::Mat_<cv::Vec4b> tmpCr;
			ModifyCrossMapArmlengthToFitSubImage(crossMapLeft(cv::Rect(x, y, w, h)), crossArmLength, tmpCr);
			subCrossMapLeft[iy] = tmpCr.clone();
		}
	}

	if (DO_RIGHT)
	{
		cf.WrapperForSkelonBuild(imRightBlur, crossArmLength, crossMapRight);
		subCrossMapRight.clear();
		subCrossMapRight.resize(numOfLabelsRight);

		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			int py, px;
			py = repPixelsRight[iy][0][0];
			px = repPixelsRight[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeRight[py][px][2]-subRangeRight[py][px][0]+1;
			int h = subRangeRight[py][px][3]-subRangeRight[py][px][1]+1;
			int x = subRangeRight[py][px][0];
			int y = subRangeRight[py][px][1];

			cv::Mat_<cv::Vec4b> tmpCr;
			ModifyCrossMapArmlengthToFitSubImage(crossMapRight(cv::Rect(x, y, w, h)), crossArmLength, tmpCr);
			subCrossMapRight[iy] = tmpCr.clone();	
		}
	}
#endif

#if USE_CLMF1_TO_AGGREGATE_COST
	int CLMF1Radius = g_filterKernelSize;
	float CLMF1Epsl = g_filterKernelEpsl/10000.0;

	if (DO_LEFT)
	{
		subCLMFLeft.clear();
		subCLMFLeft.resize(numOfLabelsLeft);

		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			subCrossMapLeft[iy].copyTo(subCLMFLeft[iy].crossMap);
			subCLMFLeft[iy].InitiateGuidance(subImageLeft[iy], subCLMFLeft[iy].crossMap, CLMF1Epsl);
		}
	}

	if (DO_RIGHT)
	{
		subCLMFRight.clear();
		subCLMFRight.resize(numOfLabelsRight);
		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			subCrossMapRight[iy].copyTo(subCLMFRight[iy].crossMap);
			subCLMFRight[iy].InitiateGuidance(subImageRight[iy], subCLMFRight[iy].crossMap, CLMF1Epsl);	
		}
	}

#endif
}

void SuperPatchmatch::ClearUpMemory()
{
	imLeftUp.release();
	imRightUp.release();
	int iy;
	if (DO_LEFT)
	{
		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			subImageLeft[iy].release();
			//		subCrossMapLeft[iy].release();
		}
		subImageLeft.clear();
		//	subCrossMapLeft.clear();
	}

	if (DO_RIGHT)
	{
		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			subImageRight[iy].release();
			//		subCrossMapRight[iy].release();
		}
		subImageRight.clear();
		//	subCrossMapRight.clear();
	}

#if USE_GF_TO_FILTER_COST
	if (DO_LEFT)
	{
		for (iy=0; iy<subGFLeft.size(); ++iy)
			subGFLeft[iy].ClearUp();
		subGFLeft.clear();
	}

	if (DO_RIGHT)
	{
		for (iy=0; iy<subGFRight.size(); ++iy)
			subGFRight[iy].ClearUp();
		subGFRight.clear();
	}

#endif


#if USE_ENHANCED_DAISY_FLOW_FEATURES
	//int iy;
	for (iy=0; iy<MAX_SCALE_LEVEL_LEFT; ++iy)
	{		
		delete descLeft[iy];
	}

	for (iy=0; iy<MAX_SCALE_LEVEL_RIGHT; ++iy)
	{
		delete descRight[iy];
	}
#endif
}

void SuperPatchmatch::GetSuperpixelsListFromSegment( const cv::Mat_<int> &segLabels, int numOfLabels, vector<vector<cv::Vec2i>> &spPixelsList )
{
	int iy, ix, height, width;
	height = segLabels.rows;
	width = segLabels.cols;

	spPixelsList.clear();
	spPixelsList.resize(numOfLabels);
	for (iy=0; iy<numOfLabels; ++iy)
		spPixelsList[iy].clear();
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmpLabel = segLabels[iy][ix];
			spPixelsList[tmpLabel].push_back(cv::Vec2i(iy, ix));
		}
	}
}

void SuperPatchmatch::RandomAssignRepresentativePixel( const vector<vector<cv::Vec2i>> &spPixelsList, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel )
{
	rePixel.create(numOfLabels, 1);
	cv::RNG rng;
	int iy;
	for (iy=0; iy<numOfLabels; ++iy)
	{
		rePixel[iy][0] = spPixelsList[iy][rng.next() % spPixelsList[iy].size()];
	}
}

void SuperPatchmatch::BuildSuperpixelsPropagationGraph( const cv::Mat_<int> &refSegLabel, int numOfLabels, const cv::Mat_<cv::Vec3f> &refImg, GraphStructure &spGraph )
{
	spGraph.adjList.clear();
	spGraph.vertexNum = 0;
	// build superpixel connectivity graph
	spGraph.ReserveSpace(numOfLabels*20);
	spGraph.SetVertexNum(numOfLabels);
	int iy, ix, height, width;
	height = refSegLabel.rows;
	width = refSegLabel.cols;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmp1 = refSegLabel[iy][ix];
			if (iy > 0) 
			{
				int tmp2 = refSegLabel[iy-1][ix];
				if (tmp1 != tmp2)
				{
					spGraph.AddEdge(tmp1, tmp2);
					//spGraph.AddEdge(tmp2, tmp1);
				}
			}

			if (ix > 0)
			{
				int tmp2 = refSegLabel[iy][ix-1];
				if (tmp1 != tmp2)
				{
					spGraph.AddEdge(tmp1, tmp2);
					//spGraph.AddEdge(tmp2, tmp1);
				}
			}
		}
	}
}


void SuperPatchmatch::ModifyCrossMapArmlengthToFitSubImage( const cv::Mat_<cv::Vec4b> &crMapIn, int maxArmLength, cv::Mat_<cv::Vec4b> &crMapOut )
{
	int iy, ix, height, width;
	height = crMapIn.rows;
	width = crMapIn.cols;
	crMapOut = crMapIn.clone();
	// up
	for (iy=0; iy<min<int>(maxArmLength, height); ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			crMapOut[iy][ix][1] = min<int>(iy, crMapOut[iy][ix][1]);
		}
	}

	// down
	int ky = maxArmLength-1;
	for (iy=height-maxArmLength; iy<height; ++iy)
	{
		if (iy < 0)
		{
			--ky;
			continue;
		}
		for (ix=0; ix<width; ++ix)
		{
			crMapOut[iy][ix][3] = min<int>(ky, crMapOut[iy][ix][3]);
		}
		--ky;
	}

	// left
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<min<int>(width, maxArmLength); ++ix)
		{
			crMapOut[iy][ix][0] = min<int>(ix, crMapOut[iy][ix][0]);
		}
	}

	// right
	int kx;
	for (iy=0; iy<height; ++iy)
	{
		kx = maxArmLength-1;
		for (ix=width-maxArmLength; ix<width; ++ix)
		{
			if (ix < 0)
			{
				--kx;
				continue;
			}
			crMapOut[iy][ix][2] = min<int>(kx, crMapOut[iy][ix][2]);
			--kx;
		}
	}
}

struct CANDIDATE_TYPE
{
	float sumWeight;
	int spId;
	CANDIDATE_TYPE(float sw, int si) : sumWeight(sw), spId(si){}
};

bool COMP_CANDIDATE_TYPE(const CANDIDATE_TYPE &ct1, const CANDIDATE_TYPE &ct2)
{
	return (ct1.sumWeight > ct2.sumWeight);
}

void SuperPatchmatch::AssociateLeftImageItselfToEstablishNonlocalPropagation( int sampleNum, int topK )
{
	const float MAX_FL = std::max<float>(horRangeLeft, verRangeLeft)/4.0;
	const float sigmaSpatial = MAX_FL*MAX_FL;
	const float sigmaColor = 25.0*25.0;
	const float MAX_SPATIAL_DISTANCE = MAX_FL*MAX_FL*6.25;
	int iLt, iRt;
	for (iLt=0; iLt<numOfLabelsLeft; ++iLt)
	{
		vector<CANDIDATE_TYPE> vecCandi;
		vecCandi.clear();
		int ltSpSize = superpixelsListLeft[iLt].size();
		for (iRt=0; iRt<numOfLabelsLeft; ++iRt)
		{
			int rtSpSize = superpixelsListLeft[iRt].size();
			float sumWeight = 0.0;
			int iy;
			for (iy=0; iy<sampleNum; ++iy)
			{
				cv::Vec2i pL = superpixelsListLeft[iLt][rand()%ltSpSize];
				cv::Vec2i pR = superpixelsListLeft[iRt][rand()%rtSpSize];
				float tmpSpatial = (pL[0]-pR[0])*(pL[0]-pR[0]) + (pL[1]-pR[1])*(pL[1]-pR[1]);
				if (tmpSpatial > MAX_SPATIAL_DISTANCE)
				{
					break;
				}
				cv::Vec3f pixL, pixR; 
				pixL = imLeft[pL[0]][pL[1]];
				pixR = imLeft[pR[0]][pR[1]];
				float tmpColor = (pixL[0]-pixR[0])*(pixL[0]-pixR[0]) 
					+ (pixL[1]-pixR[1])*(pixL[1]-pixR[1])
					+ (pixL[2]-pixR[2])*(pixL[2]-pixR[2]);
				float colorDis = exp(-tmpColor/sigmaColor);
				sumWeight += colorDis;
			}

			if (iy >= sampleNum) vecCandi.push_back(CANDIDATE_TYPE(sumWeight, iRt));
		}

		sort(vecCandi.begin(), vecCandi.end(), COMP_CANDIDATE_TYPE);

		int iy, cnt = 0;
		for (iy=0; iy<vecCandi.size(); ++iy)
		{
			if (vecCandi[iy].sumWeight < sampleNum*0.2) break;
			int tmpId = vecCandi[iy].spId;			
			// not itself
			if (tmpId != iLt)
			{
				// not in its spatial adjacency list
				std::set<int>::iterator sIt;
				std::set<int> &sAdj = spGraphLeft.adjList[iLt];
				for (sIt=sAdj.begin(); sIt!=sAdj.end(); ++sIt)
				{
					if (tmpId == *sIt) break;
				}
				if (sIt == sAdj.end())
				{
					spGraphLeft.AddEdge(iLt, tmpId);
					if (++cnt > topK) break;
				}
			}
		}
	}
}

void SuperPatchmatch::AssociateRightImageItselfToEstablishNonlocalPropagation(int sampleNum, int topK )
{
	const float MAX_FL = std::max<float>(horRangeLeft, verRangeLeft)/4.0;
	const float sigmaSpatial = MAX_FL*MAX_FL;
	const float sigmaColor = 25.0*25.0;
	const float MAX_SPATIAL_DISTANCE = MAX_FL*MAX_FL*6.25;
	int iLt, iRt;
	for (iLt=0; iLt<numOfLabelsRight; ++iLt)
	{
		vector<CANDIDATE_TYPE> vecCandi;
		vecCandi.clear();
		int ltSpSize = superpixelsListRight[iLt].size();
		for (iRt=0; iRt<numOfLabelsRight; ++iRt)
		{
			int rtSpSize = superpixelsListRight[iRt].size();
			float sumWeight = 0.0;
			int iy;
			for (iy=0; iy<sampleNum; ++iy)
			{
				cv::Vec2i pL = superpixelsListRight[iLt][rand()%ltSpSize];
				cv::Vec2i pR = superpixelsListRight[iRt][rand()%rtSpSize];
				float tmpSpatial = (pL[0]-pR[0])*(pL[0]-pR[0]) + (pL[1]-pR[1])*(pL[1]-pR[1]);
				if (tmpSpatial > MAX_SPATIAL_DISTANCE)
				{
					break;
				}
				cv::Vec3f pixL, pixR; 
				pixL = imRight[pL[0]][pL[1]];
				pixR = imRight[pR[0]][pR[1]];
				float tmpColor = (pixL[0]-pixR[0])*(pixL[0]-pixR[0]) 
					+ (pixL[1]-pixR[1])*(pixL[1]-pixR[1])
					+ (pixL[2]-pixR[2])*(pixL[2]-pixR[2]);
				float colorDis = exp(-tmpColor/sigmaColor);
				sumWeight += colorDis;
			}

			if (iy >= sampleNum) vecCandi.push_back(CANDIDATE_TYPE(sumWeight, iRt));
		}

		sort(vecCandi.begin(), vecCandi.end(), COMP_CANDIDATE_TYPE);

		int iy, cnt = 0;
		for (iy=0; iy<vecCandi.size(); ++iy)
		{
			if (vecCandi[iy].sumWeight < sampleNum*0.2) break;
			int tmpId = vecCandi[iy].spId;
			// not itself
			if (tmpId != iLt)
			{
				// not in its spatial adjacency list
				std::set<int>::iterator sIt;
				std::set<int> &sAdj = spGraphRight.adjList[iLt];
				for (sIt=sAdj.begin(); sIt!=sAdj.end(); ++sIt)
				{
					if (tmpId == *sIt) break;
				}
				if (sIt == sAdj.end())
				{
					spGraphRight.AddEdge(iLt, tmpId);
					if (++cnt > topK) break;
				}
			}
		}
	}
}

#pragma region Utility_Function

double SuperPatchmatch::CalculateSumOfSubImages( vector<cv::Mat_<cv::Vec3f>> &subImage )
{
	int iy;
	double sum = 0.0;
	for (iy=0; iy<subImage.size(); ++iy)
		sum += subImage[iy].rows*subImage[iy].cols;
	return sum;
}


inline bool SuperPatchmatch::VerifyLabelInHashLeft( int spLabel, float horLabel, float verLabel )
{
#if !DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW
	//	++spLabelVisitedNumberLeft[spLabel];
#endif
#if DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW & SUPERPIXELS_VISITED_LABELLIST
	bool *pBool = &(spLabelHashLeft[spLabel][(int)(horLabel*upPrecision+horNegFlowNumber)][(int)(verLabel*upPrecision+verNegFlowNumber)]);
	if (*pBool == true) return false;
	else 
	{
		++spLabelVisitedNumberLeft[spLabel];
		*pBool = true;
		return true;
	}
#endif
	return true;
}

inline bool SuperPatchmatch::VerifyLabelInHashRight( int spLabel, float horLabel, float verLabel )
{
#if !DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW
	//	++spLabelVisitedNumberRight[spLabel];
#endif
#if DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW & SUPERPIXELS_VISITED_LABELLIST
	bool *pBool = &(spLabelHashRight[spLabel][(int)(horLabel*upPrecision+horNegFlowNumber)][(int)(verLabel*upPrecision+verNegFlowNumber)]);
	if (*pBool == true) return false;
	else 
	{
		++spLabelVisitedNumberRight[spLabel];
		*pBool = true;
		return true;
	}
#endif
	return true;
}

double SuperPatchmatch::TestFlowEndPointError( const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow, cv::Mat_<uchar> &errorImg )
{
	int height, width, iy, ix;
	height = flowRes.rows;
	width = flowRes.cols;
	double sumError = 0.0;
	errorImg.create(height, width);
	errorImg.setTo(cv::Scalar(0));
	// const double errThresh = 1.415;
	const double errThresh = 3.5;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			//if (gtFlow[iy][ix][0]>1e9 || gtFlow[iy][ix][1]>1e9)
			// modify to account for scale change
			if (gtFlow[iy][ix][0]>1e5 || gtFlow[iy][ix][1]>1e5)
			{
				errorImg[iy][ix] = 255;
				continue;
			}
			cv::Vec2f tmpGt = gtFlow[iy][ix];
			cv::Vec2f tmpR = flowRes[iy][ix];
			double tmpError = sqrt((tmpGt[0]-tmpR[0])*(tmpGt[0]-tmpR[0])+(tmpGt[1]-tmpR[1])*(tmpGt[1]-tmpR[1]));
			sumError += tmpError;
			if (tmpError > errThresh)
			{
				errorImg[iy][ix] = 0;
			}
			else 
			{
				errorImg[iy][ix] = 255;
			}
		}
	}

	WriteOutImageResult(errorImg, WINDOW_ERROR_EE_IMG);
	return sumError/((double)height*(double)width);
}


double SuperPatchmatch::TestFlowAngularError( const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow, cv::Mat_<uchar> &errorImg )
{
	int height, width, iy, ix;
	height = flowRes.rows;
	width = flowRes.cols;
	double sumError = 0.0;
	errorImg.create(height, width);
	errorImg.setTo(cv::Scalar(0));
	const double errThresh = 5.0;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			//if (gtFlow[iy][ix][0]>1e9 || gtFlow[iy][ix][1]>1e9)
			// modify to account for scale change
			if (gtFlow[iy][ix][0]>1e5 || gtFlow[iy][ix][1]>1e5)
			{
				errorImg[iy][ix] = 255;
				continue;
			}
			cv::Vec2d tmpGt = gtFlow[iy][ix];
			cv::Vec2d tmpR = flowRes[iy][ix];

			double cosAngle = (1.0+tmpGt[0]*tmpR[0]+tmpGt[1]*tmpR[1])/(sqrt(1.0+tmpGt[0]*tmpGt[0]+tmpGt[1]*tmpGt[1])*sqrt(1.0+tmpR[0]*tmpR[0]+tmpR[1]*tmpR[1]));
			(cosAngle > 1.0)? cosAngle = 1.0: NULL;
			double tmpError = acos(cosAngle) * 180.0 * MY_PI_INV;
			// if (fabs(tmpError-180.0) < 1e-3) tmpError = 0.0;
			// above is some case about venus sequence
			sumError += tmpError;
			if (tmpError > errThresh)
			{
				errorImg[iy][ix] = 0;
			}
			else 
			{
				errorImg[iy][ix] = 255;
			}
		}
	}

	WriteOutImageResult(errorImg, WINDOW_ERROR_AE_IMG);
	return sumError/((double)height*(double)width);
}

void SuperPatchmatch::ReadFlowFile( const char *flowFile, cv::Mat_<cv::Vec2f> &flowVec, int height, int width )
{
	float *fBuffer = new float[height*width*2];
	::ReadFlowFile(fBuffer, flowFile, height, width);
	int iy, ix;
	flowVec.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			float tmp0, tmp1;
			tmp0 = fBuffer[iy*2*width+2*ix];
			tmp1 = fBuffer[iy*2*width+2*ix+1];
			//tmp0>1e9? tmp0 = -1.0: NULL;
			//tmp1>1e9? tmp1 = -1.0: NULL;
			flowVec[iy][ix][0] = tmp0;
			flowVec[iy][ix][1] = tmp1;
		}
	}

	delete [] fBuffer;
}


void SuperPatchmatch::WriteFlowFile( const char *flowFile, const cv::Mat_<cv::Vec2f> &flowVec, int height, int width )
{
	float *fBuffer = new float[height*width*2];
	int iy, ix;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			fBuffer[iy*2*width+2*ix] = flowVec[iy][ix][0];
			fBuffer[iy*2*width+2*ix+1] = flowVec[iy][ix][1];
		}
	}

	::WriteFlowFile(fBuffer, flowFile, height, width);
	delete [] fBuffer;
}


void SuperPatchmatch::ShowMotionFieldInColorCoded( const cv::Mat_<cv::Vec2f> &flowVec, const char *winName, cv::Mat_<cv::Vec3b> &flowColor, float maxMotion /*= -1.0*/ )
{
	int height, width, iy, ix;
	height = flowVec.rows;
	width = flowVec.cols;
	cv::Mat_<uchar> occMask(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// modify to account for scale change
			//if (flowVec[iy][ix][0]>1e9 || flowVec[iy][ix][1]>1e9)
			if (flowVec[iy][ix][0]>1e5 || flowVec[iy][ix][1]>1e5) occMask[iy][ix] = 255;
			else occMask[iy][ix] = 0;
		}
	}

	maxMotion = refMaxMotion;
	cv::Mat withoutOccFlow = flowVec.clone();
	withoutOccFlow.setTo(cv::Vec2f(0.0, 0.0), occMask);
	MotionToColor(withoutOccFlow, flowColor, maxMotion);

	flowColor.setTo(cv::Vec3b(0, 0, 0), occMask);
	cv::imshow(winName, flowColor);

	WriteOutImageResult(flowColor, winName);
}


void SuperPatchmatch::WriteOutImageResult( const cv::Mat &img, const char *resName, int defaultId /*= -1*/ )
{
	if (defaultId == -1)
	{
		char fileBuf[256];
		sprintf(fileBuf, "%s_%d.png", resName, processingFrameId);
		cv::imwrite(fileBuf, img);
	}
	else
	{
		char fileBuf[256];
		sprintf(fileBuf, "%s_%d.png", resName, defaultId);
		cv::imwrite(fileBuf, img);
	}
}


void SuperPatchmatch::WarpingForwardAffine( const cv::Mat_<cv::Vec4f> &flowMapping, const cv::Mat_<cv::Vec3b> &srcImg, int dstWidth, int dstHeight, cv::Mat_<cv::Vec3b> &dstImg, const float *scaleSCoef)
{
	int iy, ix, height, width;
	height = srcImg.rows;
	width = srcImg.cols;
	//printf("here 1-1\n");
	dstImg.create(dstHeight, dstWidth);
	dstImg.setTo(cv::Scalar(0.0));

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4f tmpFl = flowMapping[iy][ix];

			int ori = oriAngle[int(tmpFl[3])];
			float step = scaleSCoef[int(tmpFl[2])];
			float ssinOri = step*sin((float)ori/180.0*M_PI);
			float scosOri = step*cos((float)ori/180.0*M_PI);
			cv::Matx23f tranMat(scosOri, -ssinOri, tmpFl[0], ssinOri, scosOri, tmpFl[1]);
			cv::Matx31f p0(ix, iy, 1.0);
			cv::Matx21f p1 = tranMat*p0;
			int ny = p1.val[1];
			int nx = p1.val[0];
			if (ny<0 || ny>=dstHeight || nx<0 || nx>=dstWidth) continue;
			dstImg[ny][nx] = srcImg[iy][ix];
		}
	}
}

void SuperPatchmatch::WarpingReverseAffine( const cv::Mat_<cv::Vec4f> &flowMapping, const cv::Mat_<cv::Vec3b> &dstImg, cv::Mat_<cv::Vec3b> &srcImg, const float *scaleSCoef )
{
	int iy, ix;
	int srcHeight, srcWidth;
	srcHeight = flowMapping.rows;
	srcWidth = flowMapping.cols;
	//printf("here 2-1\n");
	srcImg.create(srcHeight, srcWidth);
	srcImg.setTo(cv::Scalar(0.0));
	int dstHeight = dstImg.rows;
	int dstWidth = dstImg.cols;
	//printf("here 2-2\n");
	for (iy=0; iy<srcHeight; ++iy)
	{
		for (ix=0; ix<srcWidth; ++ix)
		{
			/*cv::Vec2f tmpFl = flowMapping[iy][ix];
			int ny = iy+tmpFl[1];
			int nx = ix+tmpFl[0];
			if (ny<0 || ny>=dstHeight || nx<0 || nx>=dstWidth) continue;
			srcImg[iy][ix] = dstImg[ny][nx];*/

			cv::Vec4f tmpFl = flowMapping[iy][ix];

			int ori = oriAngle[int(tmpFl[3])];
			float step = scaleSCoef[int(tmpFl[2])];
			float ssinOri = step*sin((float)ori/180.0*M_PI);
			float scosOri = step*cos((float)ori/180.0*M_PI);
			cv::Matx23f tranMat(scosOri, -ssinOri, tmpFl[0], ssinOri, scosOri, tmpFl[1]);
			//cv::Matx33f invTranMat = tranMat.inv();
			cv::Matx31f p0(ix, iy, 1.0);
			cv::Matx21f p1 = tranMat*p0;
			int ny = p1.val[1];
			int nx = p1.val[0];
			if (ny<0 || ny>=dstHeight || nx<0 || nx>=dstWidth) continue;
			srcImg[iy][ix] = dstImg[ny][nx];
			// dstImg[ny][nx] = srcImg[iy][ix];
		}
	}
}


void SuperPatchmatch::ConvertAffineParametersToFlow( const cv::Mat_<cv::Vec4f> &affFlow, cv::Mat_<cv::Vec2f> &motionFlow, const float *scaleSCoef )
{
	int iy, ix, height, width;
	height = affFlow.rows;
	width = affFlow.cols;

	motionFlow.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4f tmpFl = affFlow[iy][ix];
			int ori = oriAngle[int(tmpFl[3])];
			float step = scaleSCoef[int(tmpFl[2])];
			float ssinOri = step*sin((float)ori/180.0*M_PI);
			float scosOri = step*cos((float)ori/180.0*M_PI);
			cv::Matx23f tranMat(scosOri, -ssinOri, tmpFl[0], ssinOri, scosOri, tmpFl[1]);
			cv::Matx31f p0(ix, iy, 1.0);
			cv::Matx21f p1 = tranMat*p0;

			motionFlow[iy][ix][0] = p1.val[0]-p0.val[0];
			motionFlow[iy][ix][1] = p1.val[1]-p0.val[1];
		}
	}
}


void SuperPatchmatch::CrossCheckToCreateConfidenceMask( const cv::Mat_<cv::Vec4f> &fwFlow, const cv::Mat_<cv::Vec4f> &bwFlow, float radiusThresh, cv::Mat_<uchar> &srcConfMask, cv::Mat_<uchar> &dstConfMask )
{
	int iy, ix, srcHeight, srcWidth, dstHeight, dstWidth;
	srcHeight = fwFlow.rows;
	srcWidth = fwFlow.cols;
	dstHeight = bwFlow.rows;
	dstWidth = bwFlow.cols;

	srcConfMask.create(srcHeight, srcWidth);
	srcConfMask.setTo(cv::Scalar(0.0));

	dstConfMask.create(dstHeight, dstWidth);
	dstConfMask.setTo(cv::Scalar(0.0));
	for (iy=0; iy<srcHeight; ++iy)
	{
		for (ix=0; ix<srcWidth; ++ix)
		{
			cv::Vec4f tmpFwFl = fwFlow[iy][ix];

			int ori = oriAngle[int(tmpFwFl[3])];
			float step = scaleSCoefRight[int(tmpFwFl[2])];
			float ssinOri = step*sin((float)ori/180.0*M_PI);
			float scosOri = step*cos((float)ori/180.0*M_PI);
			cv::Matx23f tranMat(scosOri, -ssinOri, tmpFwFl[0], ssinOri, scosOri, tmpFwFl[1]);
			cv::Matx31f p0(ix, iy, 1.0);
			cv::Matx21f p1 = tranMat*p0;
			int ny = p1.val[1];
			int nx = p1.val[0];
			if (ny<0 || ny>=dstHeight || nx<0 || nx>=dstWidth) continue;

			cv::Vec4f tmpBwFl = bwFlow[ny][nx];

			ori = oriAngle[int(tmpBwFl[3])];
			step = scaleSCoefLeft[int(tmpBwFl[2])];
			ssinOri = step*sin((float)ori/180.0*M_PI);
			scosOri = step*cos((float)ori/180.0*M_PI);
			cv::Matx23f bwTranMat(scosOri, -ssinOri, tmpBwFl[0], ssinOri, scosOri, tmpBwFl[1]);
			p0 = cv::Matx31f(nx, ny, 1.0);
			p1 = bwTranMat*p0;

			if ((p1.val[0]-ix)*(p1.val[0]-ix)+(p1.val[1]-iy)*(p1.val[1]-iy)<=radiusThresh*radiusThresh)
			{
				srcConfMask[iy][ix] = 255;
				dstConfMask[ny][nx] = 255;
			}
		}
	}
}


#pragma endregion

#pragma region FOR_DAISY_FLOW

#if USE_ENHANCED_DAISY_FLOW_FEATURES
void SuperPatchmatch::InitializeDaisyFeatures()
{
	//const float SCALE_INTERVAL_LEFT = 1.24f; // 1.24f
	//const float SCALE_INTERVAL_RIGHT = 1.25f; // 1.24f
	const float SCALE_LEVEL_LEFT[] = {1.0, 1.25, 1.5, 2.0, 0.5, 0.75};
	const float SCALE_LEVEL_RIGHT[] = {1.0, 1.5, 2.0, 2.5, 0.5};

	scaleSigmaLeft[0] = 0.5;
	scaleGaussSizeLeft[0] = 3;
	scaleDescRadiusLeft[0] = BASE_DESCRIPTOR_SIZE;
	scaleSCoefLeft[0] = 1.0;

	scaleSigmaRight[0] = 0.5;
	scaleGaussSizeRight[0] = 3;
	scaleDescRadiusRight[0] = BASE_DESCRIPTOR_SIZE;
	scaleSCoefRight[0] = 1.0;

	int iy;
	for (iy=1; iy<MAX_SCALE_LEVEL_LEFT; ++iy)
	{
		scaleSigmaLeft[iy] = scaleSigmaLeft[0]*SCALE_LEVEL_LEFT[iy];
		scaleGaussSizeLeft[iy] = int(scaleGaussSizeLeft[0]*SCALE_LEVEL_LEFT[iy])|1;
		scaleDescRadiusLeft[iy] = scaleDescRadiusLeft[0]*SCALE_LEVEL_LEFT[iy];
		scaleSCoefLeft[iy] = scaleSCoefLeft[0]*SCALE_LEVEL_LEFT[iy];
	}

	for (iy=1; iy<MAX_SCALE_LEVEL_RIGHT; ++iy)
	{
		scaleSigmaRight[iy] = scaleSigmaRight[0]*SCALE_LEVEL_RIGHT[iy];
		scaleGaussSizeRight[iy] = int(scaleGaussSizeRight[0]*SCALE_LEVEL_RIGHT[iy])|1;
		scaleDescRadiusRight[iy] = scaleDescRadiusRight[0]*SCALE_LEVEL_RIGHT[iy];
		scaleSCoefRight[iy] = scaleSCoefRight[0]*SCALE_LEVEL_RIGHT[iy];
	}

	if (MAX_ORIENTATION_LEVEL == 1)
	{
		oriAngle[0] = 0.0;
		oriCosTheta[0] = std::cos(oriAngle[0]/180.0*M_PI);
		oriSinTheta[0] = std::sin(oriAngle[0]/180.0*M_PI);
	}
	else // when MAX_ORIENTATION_LEVEL = 3, 5, 7, 13
	{
		float tmp = (MAX_ORIENTATION_LEVEL-1);
		float ORIENTATION_INTERVAL = 90.0/tmp;
		oriAngle[0] = -45.0;
		for (iy=1; iy<MAX_ORIENTATION_LEVEL; ++iy)
		{
			oriAngle[iy] = oriAngle[iy-1]+ORIENTATION_INTERVAL;
		}

		// daisy only accept the angle range [0, 360)
		for (iy=0; iy<(MAX_ORIENTATION_LEVEL-1)/2; ++iy)
		{
			oriAngle[iy] += 360.0;
		}

		for (iy=0; iy<MAX_ORIENTATION_LEVEL; ++iy)
		{
			oriCosTheta[iy] = std::cos(oriAngle[iy]/180.0*M_PI);
			oriSinTheta[iy] = std::sin(oriAngle[iy]/180.0*M_PI);
		}
	}


	// create blurred image and initialize daisy descriptor
	CalcTime ct;
	ct.Start();
	cv::cvtColor(imLeftOrigin, imgGrayLeft, CV_BGR2GRAY);
	cv::cvtColor(imRightOrigin, imgGrayRight, CV_BGR2GRAY);

	cv::Mat currImgGrayLeft, currImgGrayRight;
	//imgGrayLeft.copyTo(currImgGrayLeft);
	//imgGrayRight.copyTo(currImgGrayRight);
	for (iy=0; iy<MAX_SCALE_LEVEL_LEFT; ++iy)
	{
		// there could be two options: one is progressively blur image, the other is to apply on the original image
		cv::GaussianBlur(imgGrayLeft, currImgGrayLeft, cv::Size(scaleGaussSizeLeft[iy], scaleGaussSizeLeft[iy]), scaleSigmaLeft[iy], scaleSigmaLeft[iy]);
		descLeft[iy] = InitializeOneDaisyDesc(currImgGrayLeft, scaleDescRadiusLeft[iy], DAISY_FEATURE_LAYERS);

	}
	for (iy=0; iy<MAX_SCALE_LEVEL_RIGHT; ++iy)
	{
		cv::GaussianBlur(imgGrayRight, currImgGrayRight, cv::Size(scaleGaussSizeRight[iy], scaleGaussSizeRight[iy]), scaleSigmaRight[iy], scaleSigmaRight[iy]);
		descRight[iy] = InitializeOneDaisyDesc(currImgGrayRight, scaleDescRadiusRight[iy], DAISY_FEATURE_LAYERS);
	}
	ct.End("Initialize the Daisy Descriptors");

	if (DO_LEFT)
	{
		// [added] - 2013-03-20, try to initiate a sub-image buffer
		subDaisyLeft.clear();
		subDaisyLeft.resize(numOfLabelsLeft);

		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			int py, px;
			py = repPixelsLeft[iy][0][0];
			px = repPixelsLeft[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeLeft[py][px][2]-subRangeLeft[py][px][0]+1;
			int h = subRangeLeft[py][px][3]-subRangeLeft[py][px][1]+1;
			int x = subRangeLeft[py][px][0];
			int y = subRangeLeft[py][px][1];

			ExtractSubImageDaisyDescriptors(descLeft[0], subDaisyLeft[iy], y, x, scaleSCoefLeft[0], oriAngle[(MAX_ORIENTATION_LEVEL-1)/2], h, w, heightLeft, widthLeft);
		}
	}

	if (DO_RIGHT)
	{
		subDaisyRight.clear();
		subDaisyRight.resize(numOfLabelsRight);

		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			int py, px;
			py = repPixelsRight[iy][0][0];
			px = repPixelsRight[iy][0][1];
			// extract sub-image from subrange
			int w = subRangeRight[py][px][2]-subRangeRight[py][px][0]+1;
			int h = subRangeRight[py][px][3]-subRangeRight[py][px][1]+1;
			int x = subRangeRight[py][px][0];
			int y = subRangeRight[py][px][1];

			ExtractSubImageDaisyDescriptors(descRight[0], subDaisyRight[iy], y, x, scaleSCoefRight[0], oriAngle[(MAX_ORIENTATION_LEVEL-1)/2], h, w, heightRight, widthRight);
		}
	}

	ct.End("Pre-buffering the sub-images Daisy Descriptors");
}

void SuperPatchmatch::RunDaisyFilterFlow( cv::Mat_<cv::Vec2f> &flowResult)
{
#if OUTPUT_INFO_TIME
	FILE *fout = fopen("Stats.txt", "a+");
#endif
	double totalTime = 0.0;
	CalcTime ct;
	ct.Start();

	srand(time(NULL));

	imLeftOrigin.convertTo(imLeft, CV_32FC3);
	imRightOrigin.convertTo(imRight, CV_32FC3);

	widthLeft = imLeft.cols;
	heightLeft = imLeft.rows;

	widthRight = imRight.cols;
	heightRight = imRight.rows;

	// [added] - 2013-03-06, set left and right flow range
	minHorPosLeft = 0;
	maxHorPosLeft = widthLeft-1;
	horRangeLeft = maxHorPosLeft-minHorPosLeft+1;
	minVerPosLeft = 0;
	maxVerPosLeft = heightLeft-1;
	verRangeLeft = maxVerPosLeft-minVerPosLeft+1;

	minHorPosRight = 0;
	maxHorPosRight = widthRight-1;
	horRangeRight = maxHorPosRight-minHorPosRight+1;
	minVerPosRight = 0;
	maxVerPosRight = heightRight-1;
	verRangeRight = maxVerPosRight-minVerPosRight+1;

	// [added] - 2012-10-12, to randomly assign the representative pixel of each super-pixel
	if (DO_LEFT) RandomAssignRepresentativePixel(superpixelsListLeft, numOfLabelsLeft, repPixelsLeft);
	if (DO_RIGHT) RandomAssignRepresentativePixel(superpixelsListRight, numOfLabelsRight, repPixelsRight);
	// initiate advanced part
	InitiateBufferData();

	// [added] - 2013-03-20, for daisy flow
	InitializeDaisyFeatures();

	// set the sub-pixel precision level to 1
	upPrecision = 1;

	// create visited label list and initiate
	int iy, ix;
#if SUPERPIXELS_VISITED_LABELLIST
	//horFlowNumber = horFlowRange*upPrecision+1;
	//verFlowNumber = verFlowRange*upPrecision+1;

	//horNegFlowNumber = (-minHorFlow)*upPrecision;
	//verNegFlowNumber = (-minVerFlow)*upPrecision;

#if DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW
	spLabelHashLeft.resize(numOfLabelsLeft);
	for (iy=0; iy<numOfLabelsLeft; ++iy)
	{
		spLabelHashLeft[iy] = new bool *[horFlowNumber];
		for (ix=0; ix<horFlowNumber; ++ix)
		{
			spLabelHashLeft[iy][ix] = new bool [verFlowNumber];
			memset(spLabelHashLeft[iy][ix], 0, sizeof(bool)*verFlowNumber);
		}
	}

	spLabelHashRight.resize(numOfLabelsRight);
	for (iy=0; iy<numOfLabelsRight; ++iy)
	{
		spLabelHashRight[iy] = new bool *[horFlowNumber];
		for (ix=0; ix<horFlowNumber; ++ix)
		{
			spLabelHashRight[iy][ix] = new bool [verFlowNumber];
			memset(spLabelHashRight[iy][ix], 0, sizeof(bool)*verFlowNumber);
		}
	}
#endif
	spFlowVisitedNumberLeft.resize(numOfLabelsLeft);
	for (iy=0; iy<numOfLabelsLeft; ++iy) spFlowVisitedNumberLeft[iy] = 0;
	spFlowVisitedNumberRight.resize(numOfLabelsRight);
	for (iy=0; iy<numOfLabelsRight; ++iy) spFlowVisitedNumberRight[iy] = 0;

	spFlowVisitedLeft.resize(numOfLabelsLeft);
	for (iy=0; iy<numOfLabelsLeft; ++iy) spFlowVisitedLeft[iy].clear();
	spFlowVisitedRight.resize(numOfLabelsRight);
	for (iy=0; iy<numOfLabelsRight; ++iy) spFlowVisitedRight[iy].clear();
#endif

	bestLeftCost.create(heightLeft, widthLeft);

	bestRightCost.create(heightRight, widthRight);

	// initiate the flow label
	for (iy=0; iy<heightLeft; ++iy)
	{
		for (ix=0; ix<widthLeft; ++ix)
		{
			bestLeftCost[iy][ix] = DOUBLE_MAX;
		}
	}

	for (iy=0; iy<heightRight; ++iy)
	{
		for (ix=0; ix<widthRight; ++ix)
		{
			bestRightCost[iy][ix] = DOUBLE_MAX;
		}
	}

	// [added] - 2013-03-21, store the best daisy flow
	bestLeftDaisyFlow.create(heightLeft, widthLeft);
	bestRightDaisyFlow.create(heightRight, widthRight);

	// explicit initialize scale and orientation to zero to avoid error
	for (iy = 0; iy<heightLeft; ++iy)
	{
		for (ix = 0; ix<widthLeft; ++ix)
		{
			bestLeftDaisyFlow[iy][ix][2] = 0;
			bestLeftDaisyFlow[iy][ix][3] = (MAX_ORIENTATION_LEVEL - 1) / 2;
		}
	}
	for (iy = 0; iy<heightRight; ++iy)
	{
		for (ix = 0; ix<widthRight; ++ix)
		{
			bestRightDaisyFlow[iy][ix][2] = 0;
			bestRightDaisyFlow[iy][ix][3] = (MAX_ORIENTATION_LEVEL - 1) / 2;
		}
	}

	// [added] - 2012-11-07
	double tmpSubSum = CalculateSumOfSubImages(subImageLeft);
	printf("Left Sub-Image Total = %.0f, Overhead Ratio = %.4f\n", tmpSubSum, tmpSubSum/(widthLeft*heightLeft));
	tmpSubSum = CalculateSumOfSubImages(subImageRight);
	printf("Right Sub-Image Total = %.0f, Overhead Ratio = %.4f\n", tmpSubSum, tmpSubSum/(widthRight*heightRight));

	printf("Statistics: NumOfPixels Left = %d\n", widthLeft*heightLeft);
	printf("Statistics: NumOfPixels Right = %d\n", widthRight*heightRight);

	printf("Statistics: ApproximateSizeOfPixels Left = %d\n", widthLeft*heightLeft/g_spNumber);
	printf("Statistics: ApproximateSizeOfPixels Right = %d\n", widthRight*heightRight/g_spNumber);

	// [added] - 2012-11-09
	updatedLabelNumberLeft = 0.0;
	updatedLabelNumberRight = 0.0;

	if (DO_LEFT) BuildSuperpixelsPropagationGraph(segLabelsLeft, numOfLabelsLeft, imLeft, spGraphLeft);
	if (DO_RIGHT) BuildSuperpixelsPropagationGraph(segLabelsRight, numOfLabelsRight, imRight, spGraphRight);


#if USE_FLOW_NONLOCAL_PROPAGATION
	if (DO_LEFT) AssociateLeftImageItselfToEstablishNonlocalPropagation(30, 5);
	if (DO_RIGHT) AssociateRightImageItselfToEstablishNonlocalPropagation(30, 5);
#endif

	if (DO_LEFT) 
	{
		// initiate each superpixel flow
		for (iy=0; iy<numOfLabelsLeft; ++iy)
		{
			int ky, kx;
			ky = repPixelsLeft[iy][0][0];
			kx = repPixelsLeft[iy][0][1];

#if USE_ZERO_FLOW_INITIALIZATION
			if (VerifyFlowInVisitedListLeft(iy, cv::Vec4f(0, 0, 0, (MAX_ORIENTATION_LEVEL-1)/2)))
				ImproveDaisyFlowLabelListLeft(ky, kx, vector<cv::Vec4f>(1, cv::Vec4f(0, 0, 0, (MAX_ORIENTATION_LEVEL-1)/2)));
#else
			float tmpHorPos = floor((float(rand())/(RAND_MAX+1))*horRangeLeft)+minHorPosLeft;
			float tmpVerPos = floor((float(rand())/(RAND_MAX+1))*verRangeLeft)+minVerPosLeft;

			float tmpScale = rand() % MAX_SCALE_LEVEL;
			float tmpOri = rand() % MAX_ORIENTATION_LEVEL;
			float tmpHorLabel = tmpHorPos-kx;
			float tmpVerLabel = tmpVerPos-ky;

			if (VerifyFlowInVisitedListLeft(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))
				ImproveDaisyFlowLabelListLeft(ky, kx, vector<cv::Vec4f>(1, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)));
#endif
		}
	}

	if (DO_RIGHT) 
	{
		// initiate each superpixel disparity -- Right
		for (iy=0; iy<numOfLabelsRight; ++iy)
		{
			int ky, kx;
			ky = repPixelsRight[iy][0][0];
			kx = repPixelsRight[iy][0][1];
#if USE_ZERO_FLOW_INITIALIZATION
			if (VerifyFlowInVisitedListRight(iy, cv::Vec4f(0, 0, 0, (MAX_ORIENTATION_LEVEL-1)/2)))
				ImproveDaisyFlowLabelListRight(ky, kx, vector<cv::Vec4f>(1, cv::Vec4f(0, 0, 0, (MAX_ORIENTATION_LEVEL-1)/2)));
#else

			float tmpHorPos = floor((float(rand())/(RAND_MAX+1))*horRangeRight)+minHorPosRight;
			float tmpVerPos = floor((float(rand())/(RAND_MAX+1))*verRangeRight)+minVerPosRight;

			float tmpScale = rand() % MAX_SCALE_LEVEL;
			float tmpOri = rand() % MAX_ORIENTATION_LEVEL;

			float tmpHorLabel = tmpHorPos-kx;
			float tmpVerLabel = tmpVerPos-ky;

			if (VerifyFlowInVisitedListRight(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))
				ImproveDaisyFlowLabelListRight(ky, kx, vector<cv::Vec4f>(1, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)));
#endif
		}
	}


	totalTime += ct.End();
#if OUTPUT_INFO_TIME
	fprintf(fout, "Time:  %f ms\n\n", ct.End());
#endif

	ct.End("SPM: Initiation");
	printf("SPM: Finished initiating flow label\n");

	processingFrameId = 0;

	CalculateStatisticsAfterIteration();

	int iter;
	for (iter=0; iter<iterationTimes; ++iter)
	{
		// [added] - 2012-11-09, calculate actual updated label number
		updatedLabelNumberLeft = 0.0;
		updatedLabelNumberRight = 0.0;

		ct.Start();

		// [added] - 2012-10-12, to randomly assign the representative pixel of each super-pixel
		if (DO_LEFT) RandomAssignRepresentativePixel(superpixelsListLeft, numOfLabelsLeft, repPixelsLeft);
		if (DO_RIGHT) RandomAssignRepresentativePixel(superpixelsListRight, numOfLabelsRight, repPixelsRight);

		int ystart = 0, yend = numOfLabelsLeft, ychange = 1;
		if (iter%2 == 1)
		{
			ystart = numOfLabelsLeft-1; yend = -1; ychange = -1;
		}

		if (DO_LEFT) 
		{
			// traverse superpixel graph
			for (iy=ystart; iy!=yend; iy+=ychange)
			{
#if DO_DISPARITY_SORTED_LIST_FILTER
				vector<cv::Vec4f> dListVec;
				dListVec.clear();
#endif
				int refY, refX;
				refY = repPixelsLeft[iy][0][0];
				refX = repPixelsLeft[iy][0][1];
				std::set<int>::iterator sIt;
				std::set<int> &sAdj = spGraphLeft.adjList[iy];
				for (sIt=sAdj.begin(); sIt!=sAdj.end(); ++sIt)
				{
					// [added] - 2012-10-29
					repPixelsLeft[*sIt][0] = superpixelsListLeft[*sIt][rand()%superpixelsListLeft[*sIt].size()];
					int ky, kx;
					ky = repPixelsLeft[*sIt][0][0];
					kx = repPixelsLeft[*sIt][0][1];

					cv::Vec4f tmpFlow = bestLeftDaisyFlow[ky][kx];

#if DO_DISPARITY_SORTED_LIST_FILTER
					if (VerifyFlowInVisitedListLeft(iy, tmpFlow)) dListVec.push_back(tmpFlow);

#endif
#if DO_DISPARITY_SINGLE_FILTER
					if (VerifyFlowInVisitedListLeft(iy, tmpFlow))
						ImproveDaisyFlowLabelListLeft(refY, refX, vector<cv::Vec4f>(1, tmpFlow));
#endif
				}

				/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
				const float randomRatio = 2.0;
				float mag = std::min<float>(horRangeLeft, verRangeLeft);
				float oriMag = MAX_ORIENTATION_LEVEL*2.0;
				float scaleMag = MAX_SCALE_LEVEL_RIGHT*2.0;
				cv::Vec4f tmpFlow = bestLeftDaisyFlow[refY][refX];
				for (; mag>=(1.0/upPrecision); mag/=randomRatio, oriMag/=ORIENTATION_SAMPLE_RATIO, scaleMag/=SCALE_SAMPLE_RATIO)
				{
					float deltaHorLabel = (float(rand())/RAND_MAX-0.5)*2.0*mag;
					float deltaVerLabel = (float(rand())/RAND_MAX-0.5)*2.0*mag;

					float tmpHorLabel = tmpFlow[0]+deltaHorLabel;
					float tmpVerLabel = tmpFlow[1]+deltaVerLabel;

					tmpHorLabel = floor(tmpHorLabel*upPrecision+0.5)/upPrecision;
					tmpVerLabel = floor(tmpVerLabel*upPrecision+0.5)/upPrecision;

					float tmpScale = int(tmpFlow[2]+(float(rand())/RAND_MAX-0.5)*2.0*scaleMag+(MAX_SCALE_LEVEL_RIGHT<<2)) % MAX_SCALE_LEVEL_RIGHT;
					float tmpOri = int(tmpFlow[3]+(float(rand())/RAND_MAX-0.5)*2.0*oriMag+(MAX_ORIENTATION_LEVEL<<2)) % MAX_ORIENTATION_LEVEL;

					int ori = oriAngle[int(tmpOri)];
					float step = scaleSCoefRight[int(tmpScale)];
					float ssinOri = step*oriSinTheta[int(tmpOri)];
					float scosOri = step*oriCosTheta[int(tmpOri)];
					cv::Matx23f tranMat(scosOri, -ssinOri, tmpHorLabel, ssinOri, scosOri, tmpVerLabel);
					cv::Matx31f p0(refX, refY, 1.0);
					cv::Matx21f p1 = tranMat*p0;
					int ny = p1.val[1];
					int nx = p1.val[0];
					if (ny<minVerPosRight || ny>maxVerPosRight || nx<minHorPosRight || nx>maxHorPosRight) continue;
#if DO_DISPARITY_SORTED_LIST_FILTER
					if (VerifyFlowInVisitedListLeft(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))
						dListVec.push_back(cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri));
#endif

#if DO_DISPARITY_SINGLE_FILTER
					if (VerifyFlowInVisitedListLeft(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))
						ImproveDaisyFlowLabelListLeft(refY, refX, vector<cv::Vec4f>(1, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)));
#endif
				}

#if DO_DISPARITY_SORTED_LIST_FILTER
				// make a disparity label list and try to improve them
				if (dListVec.size() > 0) ImproveDaisyFlowLabelListLeft(refY, refX, dListVec);
#endif
			}
		}


		ystart = 0, yend = numOfLabelsRight, ychange = 1;
		if (iter%2 == 1)
		{
			ystart = numOfLabelsRight-1; yend = -1; ychange = -1;
		}

		if (DO_RIGHT)
		{
			// traverse superpixel graph
			for (iy=ystart; iy!=yend; iy+=ychange)
			{
#if DO_DISPARITY_SORTED_LIST_FILTER
				vector<cv::Vec4f> dListVec;
				dListVec.clear();
#endif
				int refY, refX;
				refY = repPixelsRight[iy][0][0];
				refX = repPixelsRight[iy][0][1];
				std::set<int>::iterator sIt;
				std::set<int> &sAdj = spGraphRight.adjList[iy];
				for (sIt=sAdj.begin(); sIt!=sAdj.end(); ++sIt)
				{
					// [added] - 2012-10-29
					repPixelsRight[*sIt][0] = superpixelsListRight[*sIt][rand()%superpixelsListRight[*sIt].size()];
					int ky, kx;
					ky = repPixelsRight[*sIt][0][0];
					kx = repPixelsRight[*sIt][0][1];

					cv::Vec4f tmpFlow = bestRightDaisyFlow[ky][kx];

#if DO_DISPARITY_SORTED_LIST_FILTER
					if (VerifyFlowInVisitedListRight(iy, tmpFlow))		
						dListVec.push_back(tmpFlow);
#endif
#if DO_DISPARITY_SINGLE_FILTER
					if (VerifyFlowInVisitedListRight(iy, tmpFlow))		
						ImproveDaisyFlowLabelListRight(refY, refX, vector<cv::Vec4f>(1, tmpFlow));
#endif
				}

				/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
				const float randomRatio = 2.0;
				float mag = std::min<float>(horRangeRight, verRangeRight);
				float oriMag = MAX_ORIENTATION_LEVEL*2.0;
				float scaleMag = MAX_SCALE_LEVEL_LEFT*2.0;
				cv::Vec4f tmpFlow = bestRightDaisyFlow[refY][refX];
				for (; mag>=(1.0/upPrecision); mag/=randomRatio, oriMag/=ORIENTATION_SAMPLE_RATIO, scaleMag/=SCALE_SAMPLE_RATIO)
				{
					float deltaHorLabel = (float(rand())/RAND_MAX-0.5)*2.0*mag;
					float deltaVerLabel = (float(rand())/RAND_MAX-0.5)*2.0*mag;

					float tmpHorLabel = tmpFlow[0]+deltaHorLabel;
					float tmpVerLabel = tmpFlow[1]+deltaVerLabel;

					tmpHorLabel = floor(tmpHorLabel*upPrecision+0.5)/upPrecision;
					tmpVerLabel = floor(tmpVerLabel*upPrecision+0.5)/upPrecision;

					float tmpScale = int(tmpFlow[2]+(float(rand())/RAND_MAX-0.5)*2.0*scaleMag+(MAX_SCALE_LEVEL_LEFT<<2)) % MAX_SCALE_LEVEL_LEFT;
					float tmpOri = int(tmpFlow[3]+(float(rand())/RAND_MAX-0.5)*2.0*oriMag+(MAX_ORIENTATION_LEVEL<<2)) % MAX_ORIENTATION_LEVEL;

					int ori = oriAngle[int(tmpOri)];
					float step = scaleSCoefLeft[int(tmpScale)];
					float ssinOri = step*oriSinTheta[int(tmpOri)];
					float scosOri = step*oriCosTheta[int(tmpOri)];
					cv::Matx23f tranMat(scosOri, -ssinOri, tmpHorLabel, ssinOri, scosOri, tmpVerLabel);
					cv::Matx31f p0(refX, refY, 1.0);
					cv::Matx21f p1 = tranMat*p0;
					int ny = p1.val[1];
					int nx = p1.val[0];
					if (ny<minVerPosLeft || ny>maxVerPosLeft || nx<minHorPosLeft || nx>maxHorPosLeft) continue;
#if DO_DISPARITY_SORTED_LIST_FILTER
					if (VerifyFlowInVisitedListRight(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))		
						dListVec.push_back(cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri));
#endif

#if DO_DISPARITY_SINGLE_FILTER
					if (VerifyFlowInVisitedListRight(iy, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)))	
						ImproveDaisyFlowLabelListRight(refY, refX, vector<cv::Vec4f>(1, cv::Vec4f(tmpHorLabel, tmpVerLabel, tmpScale, tmpOri)));
#endif
					//}
				}

#if DO_DISPARITY_SORTED_LIST_FILTER
				// make a disparity label list and try to improve them
				if (dListVec.size() > 0) ImproveDaisyFlowLabelListRight(refY, refX, dListVec);
#endif
			}
		}
		totalTime += ct.End();

#if OUTPUT_INFO_TIME
		fprintf(fout, "Time:  %f\n\n", ct.End());
#endif

		ct.End("SPM: One iteration");
		printf("SPM: Finished iteration %d\n", iter);

		++processingFrameId;
		CalculateStatisticsAfterIteration();
	}


	// write out the result
	ConvertAffineParametersToFlow(bestLeftDaisyFlow, flowResult, scaleSCoefRight);

	printf("SPM: Finished writeout result\n==================================================\n");
	printf("TotalTime = %.6f ms\n==================================================\n", totalTime);

	// [added] - try to avoid insuffcient memory error
	ClearUpMemory();
#if OUTPUT_INFO_TIME
	fprintf(fout, "====================================================\n");
	fclose(fout);
#endif

#if SUPERPIXELS_VISITED_LABELLIST & DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW
	//#if DO_USE_HASHTABLE_TO_RECORD_VISTED_FLOW
	//delete [] spHorLabelList;
	//delete [] spVerLabelList;

	for (iy=0; iy<numOfLabelsLeft; ++iy)
	{
		for (ix=0; ix<horFlowNumber; ++ix)
		{
			delete [] spLabelHashLeft[iy][ix];
		}
		delete [] spLabelHashLeft[iy];
	}

	for (iy=0; iy<numOfLabelsRight; ++iy)
	{
		for (ix=0; ix<horFlowNumber; ++ix)
		{
			delete [] spLabelHashRight[iy][ix];
		}
		delete [] spLabelHashRight[iy];
	}
#endif
}

void SuperPatchmatch::ImproveDaisyFlowLabelListLeft( int py, int px, vector<cv::Vec4f> &flowList )
{
	int pLabel = segLabelsLeft[py][px];
	int dSize = flowList.size();
#if USE_COLOR_FEATURES
	cv::Mat_<cv::Vec3f> subLt = subImageLeft[pLabel];

	int upHeight, upWidth;
	upHeight = imRightUp.rows;
	upWidth = imRightUp.cols;

	// extract sub-image from subrange
	int w = subRangeLeft[py][px][2]-subRangeLeft[py][px][0]+1;
	int h = subRangeLeft[py][px][3]-subRangeLeft[py][px][1]+1;
	int x = subRangeLeft[py][px][0];
	int y = subRangeLeft[py][px][1];


	int kd;
	cv::Mat_<float> rawCost;
	rawCost.create(h, w*dSize);

	int kx;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec2f fl = flowList[kd];

		/*int ty, tx;
		ty = y+fl[1];
		tx = x+fl[0];
		if (ty<0 || ty>heightRight || tx<0 || tx>widthRight)
		{
		rawCost(cv::Rect(kx, 0, w, h)).setTo(1e6);
		continue;
		}*/

		cv::Mat_<cv::Vec3f> subRt;
		subRt.create(h, w);

		int cy, cx, oy, ox;
		oy = y;
		for (cy=0; cy<h; ++cy, ++oy)
		{
			ox = x;
			for (cx=0; cx<w; ++cx, ++ox)
			{
				int oyUp, oxUp;
				oyUp = (oy+fl[1])*upPrecision;
				oxUp = (ox+fl[0])*upPrecision;

				(oyUp < 0)? oyUp = 0: NULL;
				(oyUp >= upHeight)? oyUp = upHeight-1: NULL;
				(oxUp < 0)? oxUp = 0: NULL;
				(oxUp >= upWidth)? oxUp = upWidth-1: NULL;

				subRt[cy][cx] = imRightUp[oyUp][oxUp];
			}
		}
		// calculate raw cost
		int iy, ix;
		for (iy=0; iy<h; ++iy)
		{
			for (ix=0; ix<w; ++ix)
			{
				float colorCost = (subLt[iy][ix][0]-subRt[iy][ix][0])*(subLt[iy][ix][0]-subRt[iy][ix][0])
					+ (subLt[iy][ix][1]-subRt[iy][ix][1])*(subLt[iy][ix][1]-subRt[iy][ix][1])
					+ (subLt[iy][ix][2]-subRt[iy][ix][2])*(subLt[iy][ix][2]-subRt[iy][ix][2]);

				rawCost[iy][kx+ix] = sqrt(colorCost);
			}
		}
	}
#endif

#if USE_ENHANCED_DAISY_FLOW_FEATURES
	cv::Mat_<float> subLt = subDaisyLeft[pLabel];

	//int upHeight, upWidth;
	//upHeight = imRightUp.rows;
	//upWidth = imRightUp.cols;

	// extract sub-image from subrange
	int w = subRangeLeft[py][px][2]-subRangeLeft[py][px][0]+1;
	int h = subRangeLeft[py][px][3]-subRangeLeft[py][px][1]+1;
	int x = subRangeLeft[py][px][0];
	int y = subRangeLeft[py][px][1];

	for (int fet=0; fet<w*h; ++fet)
	{
		for (int subLine=0; subLine<DAISY_FEATURE_LENGTH*sizeof(float); subLine+=64)
		{
			_mm_prefetch(((char *)subLt[fet])+subLine, _MM_HINT_T0);
		}
	}

	int kd;
	cv::Mat_<float> rawCost;
	rawCost.create(h, w*dSize);

	int kx;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec4f fl = flowList[kd];

		cv::Mat_<float> localRc = rawCost(cv::Rect(kx, 0, w, h)); 
		int scaleId = int(fl[2]);
		int oriId = int(fl[3]);			
		NewlyExtractAndComputeSubImageDaisyDescriptorsCost(descRight[scaleId], localRc, subLt,
			y, x, h, w, fl[1], fl[0], scaleSCoefRight[scaleId], oriAngle[oriId]);

#if USE_TRUNCATED_L2_DISTANCE
		for (int cy=0; cy<h; ++cy)
		{
			for (int cx=0; cx<w; ++cx)
			{
				(localRc[cy][cx] > TRUNCATED_L2_THRESHOLD)? localRc[cy][cx] = TRUNCATED_L2_THRESHOLD: NULL; 
			}
		}
#endif

#if USE_TRUNCATED_L1_DISTANCE
		for (int cy=0; cy<h; ++cy)
		{
			for (int cx=0; cx<w; ++cx)
			{
				(localRc[cy][cx] > TRUNCATED_L1_THRESHOLD)? localRc[cy][cx] = TRUNCATED_L1_THRESHOLD: NULL; 
			}
		}
#endif

	}

#endif
#if USE_CLMF0_TO_AGGREGATE_COST
	cv::Mat_<cv::Vec4b> leftCombinedCrossMap;
	leftCombinedCrossMap.create(h, w*dSize);
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		subCrossMapLeft[pLabel].copyTo(leftCombinedCrossMap(cv::Rect(kx, 0, w, h)));
	}

	// aggregate cost
	cv::Mat_<float> filteredCostLeft;
	CFFilter cff;

	//cff.FastCLMF0FloatFilter(leftCombinedCrossMap, rawCost, filteredCostLeft);
	cff.FastCLMF0FloatFilterPointer(leftCombinedCrossMap, rawCost, filteredCostLeft);
#endif


#if USE_GF_TO_FILTER_COST
	cv::Mat_<float> filteredCostLeft(h, w*dSize);
	GFilter *gf = &subGFLeft[pLabel];
	int gfRadius = g_filterKernelSize;
	float gfEpsl = g_filterKernelEpsl/10000.0;
	//gf.InitiateGuidance(subLt, gfRadius, gfEpsl);
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		//gf->DoGuidedFilter(rawCost(cv::Rect(kx, 0, w, h)), gfRadius, gfEpsl, tmpCost);
		gf->NewDoGuidedFilter(rawCost(cv::Rect(kx, 0, w, h)), gfRadius, gfEpsl, tmpCost);
		tmpCost.copyTo(filteredCostLeft(cv::Rect(kx, 0, w, h)));
	}
#endif

#if USE_CLMF1_TO_AGGREGATE_COST
	cv::Mat_<float> filteredCostLeft(h, w*dSize);
	CLMFilter *clmf = &subCLMFLeft[pLabel];
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		clmf->DoCLMF1Filter(rawCost(cv::Rect(kx, 0, w, h)), clmf->crossMap, tmpCost);
		tmpCost.copyTo(filteredCostLeft(cv::Rect(kx, 0, w, h)));
	}
#endif

#if USE_BOXFILTER_TO_AGGREGATE_COST
	cv::Mat_<float> filteredCostLeft(h, w*dSize);
	CostBoxFilter cbf;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		cbf.DoCostBoxFilter(rawCost(cv::Rect(kx, 0, w, h)), 3, tmpCost);
		tmpCost.copyTo(filteredCostLeft(cv::Rect(kx, 0, w, h)));
	}
#endif

#if NO_AGGREGATE_COST
	cv::Mat_<float> filteredCostLeft(h, w*dSize);
	rawCost.copyTo(filteredCostLeft);
#endif

	// update best cost and best label
	int spw = spRangeLeft[py][px][2]-spRangeLeft[py][px][0]+1;
	int sph = spRangeLeft[py][px][3]-spRangeLeft[py][px][1]+1;
	int spx = spRangeLeft[py][px][0];
	int spy = spRangeLeft[py][px][1];

	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec4f fl = flowList[kd];

		int ori = oriAngle[int(fl[3])];
		float step = scaleSCoefRight[int(fl[2])];
		float ssinOri = step*sin((float)ori/180.0*M_PI);
		float scosOri = step*cos((float)ori/180.0*M_PI);
		cv::Matx23f tranMat(scosOri, -ssinOri, fl[0], ssinOri, scosOri, fl[1]);

		// position in superpixel image
		int iy, ix;
		// position in original image
		int oy, ox;
		// position in sub-image which has a kernel size boundary around superpixel image
		int sy, sx;

		oy = spy;
		sy = spy-y;
		for (iy=0; iy<sph; ++iy, ++oy, ++sy)
		{
			ox = spx;
			sx = spx-x;
			for (ix=0; ix<spw; ++ix, ++ox, ++sx)
			{
				float tmp = filteredCostLeft[sy][kx+sx];
				if (tmp < bestLeftCost[oy][ox])
				{
					cv::Matx31f p0(ox, oy, 1.0);
					cv::Matx21f p1 = tranMat*p0;

					if (p1.val[1]<minVerPosRight || p1.val[1]>maxVerPosRight || p1.val[0]<minHorPosRight || p1.val[0]>maxHorPosRight) continue;

					bestLeftCost[oy][ox] = tmp;
					bestLeftDaisyFlow[oy][ox] = fl;
					//[added] - 2012-11-09
					++updatedLabelNumberLeft;
				}

			}
		}
	}
}

void SuperPatchmatch::ImproveDaisyFlowLabelListRight( int py, int px, vector<cv::Vec4f> &flowList )
{
	int pLabel = segLabelsRight[py][px];
	int dSize = flowList.size();
#if USE_COLOR_FEATURES
	cv::Mat_<cv::Vec3f> subRt = subImageRight[pLabel];

	int upHeight, upWidth;
	upHeight = imLeftUp.rows;
	upWidth = imLeftUp.cols;

	// extract sub-image from subrange
	int w = subRangeRight[py][px][2]-subRangeRight[py][px][0]+1;
	int h = subRangeRight[py][px][3]-subRangeRight[py][px][1]+1;
	int x = subRangeRight[py][px][0];
	int y = subRangeRight[py][px][1];


	int kd;
	cv::Mat_<float> rawCost;
	rawCost.create(h, w*dSize);

	int kx;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec2f fl = flowList[kd];

		// update label list
		// spLabelList[pLabel].insert(d);

		cv::Mat_<cv::Vec3f> subLt;
		subLt.create(h, w);

		int cy, cx, oy, ox;
		oy = y;
		for (cy=0; cy<h; ++cy, ++oy)
		{
			ox = x;
			for (cx=0; cx<w; ++cx, ++ox)
			{
				int oyUp, oxUp;
				oyUp = (oy+fl[1])*upPrecision;
				oxUp = (ox+fl[0])*upPrecision;

				(oyUp < 0)? oyUp = 0: NULL;
				(oyUp >= upHeight)? oyUp = upHeight-1: NULL;
				(oxUp < 0)? oxUp = 0: NULL;
				(oxUp >= upWidth)? oxUp = upWidth-1: NULL;

				subLt[cy][cx] = imLeftUp[oyUp][oxUp];
			}
		}

		// calculate raw cost
		int iy, ix;
		for (iy=0; iy<h; ++iy)
		{
			for (ix=0; ix<w; ++ix)
			{
				float colorCost = (subLt[iy][ix][0]-subRt[iy][ix][0])*(subLt[iy][ix][0]-subRt[iy][ix][0])
					+ (subLt[iy][ix][1]-subRt[iy][ix][1])*(subLt[iy][ix][1]-subRt[iy][ix][1])
					+ (subLt[iy][ix][2]-subRt[iy][ix][2])*(subLt[iy][ix][2]-subRt[iy][ix][2]);

				rawCost[iy][kx+ix] = sqrt(colorCost);
			}
		}
	}
#endif

#if USE_ENHANCED_DAISY_FLOW_FEATURES
	cv::Mat_<float> subRt = subDaisyRight[pLabel];

	//int upHeight, upWidth;
	//upHeight = imLeftUp.rows;
	//upWidth = imLeftUp.cols;

	// extract sub-image from subrange
	int w = subRangeRight[py][px][2]-subRangeRight[py][px][0]+1;
	int h = subRangeRight[py][px][3]-subRangeRight[py][px][1]+1;
	int x = subRangeRight[py][px][0];
	int y = subRangeRight[py][px][1];

	for (int fet=0; fet<w*h; ++fet)
	{
		for (int subLine=0; subLine<DAISY_FEATURE_LENGTH*sizeof(float); subLine+=64)
		{
			_mm_prefetch(((char *)subRt[fet])+subLine, _MM_HINT_T0);
		}
	}


	int kd;
	cv::Mat_<float> rawCost;
	rawCost.create(h, w*dSize);

	int kx;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec4f fl = flowList[kd];

		// update label list
		// spLabelList[pLabel].insert(d);
		cv::Mat_<float> localRc = rawCost(cv::Rect(kx, 0, w, h));

		int scaleId = int(fl[2]);
		int oriId = int(fl[3]);

		NewlyExtractAndComputeSubImageDaisyDescriptorsCost(descLeft[scaleId], localRc, subRt, 
			y, x, h, w, fl[1], fl[0], scaleSCoefLeft[scaleId], oriAngle[oriId]);

#if USE_TRUNCATED_L2_DISTANCE
		for (int cy=0; cy<h; ++cy)
		{
			for (int cx=0; cx<w; ++cx)
			{
				(localRc[cy][cx] > TRUNCATED_L2_THRESHOLD)? localRc[cy][cx] = TRUNCATED_L2_THRESHOLD: NULL;
			}
		}
#endif

#if USE_TRUNCATED_L1_DISTANCE
		for (int cy=0; cy<h; ++cy)
		{
			for (int cx=0; cx<w; ++cx)
			{
				(localRc[cy][cx] > TRUNCATED_L1_THRESHOLD)? localRc[cy][cx] = TRUNCATED_L1_THRESHOLD: NULL;
			}
		}
#endif

	}
#endif


#if USE_CLMF0_TO_AGGREGATE_COST
	cv::Mat_<cv::Vec4b> rightCombinedCrossMap;
	rightCombinedCrossMap.create(h, w*dSize);
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		subCrossMapRight[pLabel].copyTo(rightCombinedCrossMap(cv::Rect(kx, 0, w, h)));
	}

	// aggregate cost
	cv::Mat_<float> filteredCostRight;
	CFFilter cff;

	cff.FastCLMF0FloatFilterPointer(rightCombinedCrossMap, rawCost, filteredCostRight);
	//cff.FastCLMF0FloatFilter(rightCombinedCrossMap, rawCost, filteredCostRight);
#endif

#if USE_GF_TO_FILTER_COST
	cv::Mat_<float> filteredCostRight(h, w*dSize);
	GFilter *gf = &subGFRight[pLabel];
	int gfRadius = g_filterKernelSize;
	float gfEpsl = g_filterKernelEpsl/10000.0;
	//gf.InitiateGuidance(subRt, gfRadius, gfEpsl);
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		//gf->DoGuidedFilter(rawCost(cv::Rect(kx, 0, w, h)), gfRadius, gfEpsl, tmpCost);
		gf->NewDoGuidedFilter(rawCost(cv::Rect(kx, 0, w, h)), gfRadius, gfEpsl, tmpCost);
		tmpCost.copyTo(filteredCostRight(cv::Rect(kx, 0, w, h)));
	}
#endif

#if USE_CLMF1_TO_AGGREGATE_COST
	cv::Mat_<float> filteredCostRight(h, w*dSize);
	CLMFilter *clmf = &subCLMFRight[pLabel];
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		clmf->DoCLMF1Filter(rawCost(cv::Rect(kx, 0, w, h)), clmf->crossMap, tmpCost);
		tmpCost.copyTo(filteredCostRight(cv::Rect(kx, 0, w, h)));
	}
#endif

#if USE_BOXFILTER_TO_AGGREGATE_COST
	cv::Mat_<float> filteredCostRight(h, w*dSize);
	CostBoxFilter cbf;
	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Mat_<float> tmpCost;
		cbf.DoCostBoxFilter(rawCost(cv::Rect(kx, 0, w, h)), 3, tmpCost);
		tmpCost.copyTo(filteredCostRight(cv::Rect(kx, 0, w, h)));
	}
#endif

#if NO_AGGREGATE_COST
	cv::Mat_<float> filteredCostRight(h, w*dSize);
	rawCost.copyTo(filteredCostRight);
#endif
	// update best cost and best label
	int spw = spRangeRight[py][px][2]-spRangeRight[py][px][0]+1;
	int sph = spRangeRight[py][px][3]-spRangeRight[py][px][1]+1;
	int spx = spRangeRight[py][px][0];
	int spy = spRangeRight[py][px][1];


	for (kd=0; kd<dSize; ++kd)
	{
		kx = kd*w;
		cv::Vec4f fl = flowList[kd];

		int ori = oriAngle[int(fl[3])];
		float step = scaleSCoefLeft[int(fl[2])];
		float ssinOri = step*sin((float)ori/180.0*M_PI);
		float scosOri = step*cos((float)ori/180.0*M_PI);
		cv::Matx23f tranMat(scosOri, -ssinOri, fl[0], ssinOri, scosOri, fl[1]);

		// position in superpixel image
		int iy, ix;
		// position in original image
		int oy, ox;
		// position in sub-image which has a kernel size boundary around superpixel image
		int sy, sx;

		oy = spy;
		sy = spy-y;
		for (iy=0; iy<sph; ++iy, ++oy, ++sy)
		{
			ox = spx;
			sx = spx-x;
			for (ix=0; ix<spw; ++ix, ++ox, ++sx)
			{
				float tmp = filteredCostRight[sy][kx+sx];
				if (tmp < bestRightCost[oy][ox])
				{
					//cv::Vec4f tmpFl = fl;

					cv::Matx31f p0(ox, oy, 1.0);
					cv::Matx21f p1 = tranMat*p0;

					if (p1.val[1]<minVerPosLeft || p1.val[1]>maxVerPosLeft || p1.val[0]<minHorPosLeft || p1.val[0]>maxHorPosLeft) continue;

					bestRightCost[oy][ox] = tmp;
					//bestRightLabel[oy][ox] = fl;
					bestRightDaisyFlow[oy][ox] = fl;
					//[added] - 2012-11-09
					++updatedLabelNumberRight;
				}
			}
		}
	}
}

void SuperPatchmatch::CopyFirstTwoChannelsToFlow( const cv::Mat_<cv::Vec4f> &flIn, cv::Mat_<cv::Vec2f> &flOut )
{
	int iy, ix;
	int height = flIn.rows;
	int width = flIn.cols;
	flOut.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			flOut[iy][ix][0] = flIn[iy][ix][0];
			flOut[iy][ix][1] = flIn[iy][ix][1];
		}
	}
}


void SuperPatchmatch::CopySelectedChannelToFloat( const cv::Mat_<cv::Vec4f> &floatIn, int selCh, cv::Mat_<float> &floatOut )
{
	int iy, ix;
	int height = floatIn.rows;
	int width = floatIn.cols;
	floatOut.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			floatOut[iy][ix] = floatIn[iy][ix][selCh];
		}
	}
}

#endif

void SuperPatchmatch::TransferMaskUsingFlow( const cv::Mat_<cv::Vec3b> &maskIn, const cv::Mat_<cv::Vec2f> &motionFlow, cv::Mat_<cv::Vec3b> &maskOut )
{
	int iy, ix, dstHeight, dstWidth;
	dstHeight = maskIn.rows;
	dstWidth = maskIn.cols;
	int srcHeight, srcWidth;
	srcHeight = motionFlow.rows;
	srcWidth = motionFlow.cols;
	maskOut.create(srcHeight, srcWidth);
	maskOut.setTo(cv::Scalar(0.0));
	for (iy=0; iy<srcHeight; ++iy)
	{
		for (ix=0; ix<srcWidth; ++ix)
		{
			cv::Vec2f fl = motionFlow[iy][ix];
			int ny = int(fl[1]+iy);
			int nx = int(fl[0]+ix);
			if (ny>=dstHeight || ny<0 || nx>=dstWidth || nx<0) continue;
			maskOut[iy][ix] = maskIn[ny][nx];
		}
	}
}


void SuperPatchmatch::CreateMaskedImage( const cv::Mat_<cv::Vec3b> &imgIn, const cv::Mat_<uchar> &mask, cv::Mat_<cv::Vec3b> &imgOut )
{
	int height, width;
	height = imgIn.rows;
	width = imgIn.cols;
	imgOut.create(height, width);
	int iy, ix;
	float alphaVal = 123.0f/255.0;

	cv::Mat_<uchar> edgeMask;
	cv::morphologyEx(mask, edgeMask, cv::MORPH_GRADIENT, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			if (mask[iy][ix] == 255)
			{
				if (edgeMask[iy][ix] == 0) imgOut[iy][ix] = imgIn[iy][ix];
				else imgOut[iy][ix] = (1.0-alphaVal)*imgIn[iy][ix];
			}
			else 
			{
				if (edgeMask[iy][ix] == 255) imgOut[iy][ix] = (1.0-alphaVal)*imgIn[iy][ix];
				else imgOut[iy][ix] = alphaVal*cv::Vec3b(255.0, 255.0, 255.0)+(1.0-alphaVal)*imgIn[iy][ix];
			}
		}
	}
}



bool SuperPatchmatch::VerifyFlowInVisitedListLeft( int spLabel, const cv::Vec4f &fl )
{
	int iy;
	vector<cv::Vec4f> &spFl = spFlowVisitedLeft[spLabel];
	for (iy=0; iy<spFl.size(); ++iy)
	{
		int ix;
		bool inFlag = true;
		for (ix=0; ix<4; ++ix)
		{
			if (fabs(spFl[iy][ix]-fl[ix])>EPS) inFlag = false;
		}
		if (inFlag) return false;
	}
	spFl.push_back(fl);
	return true;
}

bool SuperPatchmatch::VerifyFlowInVisitedListRight( int spLabel, const cv::Vec4f &fl )
{
	int iy;
	vector<cv::Vec4f> &spFl = spFlowVisitedRight[spLabel];
	for (iy=0; iy<spFl.size(); ++iy)
	{
		int ix;
		bool inFlag = true;
		for (ix=0; ix<4; ++ix)
		{
			if (fabs(spFl[iy][ix]-fl[ix])>EPS) inFlag = false;
		}
		if (inFlag) return false;
	}
	spFl.push_back(fl);
	return true;
}


#pragma endregion 

void SuperPatchmatch::CalculateScaledGroundTruthFlow( const cv::Mat_<cv::Vec2f> &flowIn, cv::Mat_<cv::Vec2f> &flowOut )
{
	int hIn, wIn, iy, ix;
	hIn = flowIn.rows; 
	wIn = flowIn.cols;

	int hOut, wOut;
	hOut = floor(scaleLeftFactor*hIn+0.5);
	wOut = floor(scaleLeftFactor*wIn+0.5);

	flowOut.create(hOut, wOut);
	for (iy=0; iy<hOut; ++iy)
	{
		for (ix=0; ix<wOut; ++ix)
		{
			// assume the scale factor is less than 1.0
			float tx = (ix/scaleLeftFactor);
			float ty = (iy/scaleLeftFactor);
			int indY = max<int>(0, min<int>(hOut-1, floor(ty+0.5)));
			int indX = max<int>(0, min<int>(wOut-1, floor(tx+0.5)));
			flowOut[iy][ix][0] = (tx+flowIn[indY][indX][0])*scaleRightFactor-ix;
			flowOut[iy][ix][1] = (ty+flowIn[indY][indX][1])*scaleRightFactor-iy;
		}
	}
}

void SuperPatchmatch::PostRefineUsingBilateralFilterAsGuidance( const cv::Mat_<cv::Vec2f> &flowVec, const cv::Mat_<cv::Vec3b> &weightColorImg, int radius, float bfSigmaSpatial, float bfSigmaColor, cv::Mat_<cv::Vec2f> &refinedFlow )
{
	int height, width, iy, ix;
	height = weightColorImg.rows;
	width = weightColorImg.cols;

	int winSize = radius*2+1;
	cv::Mat_<float> spatialWeight(winSize, winSize);
	const float sigmaSpatial = bfSigmaSpatial;
	const int COLOR_DIFF_SIZE = 50*50*10;
	float colorDiffWeight[COLOR_DIFF_SIZE];	
	const float sigmaColor = bfSigmaColor;

	for (iy=-radius; iy<=radius; ++iy)
	{
		for (ix=-radius; ix<=radius; ++ix)
		{
			if (abs(iy)>7 || abs(ix)>7) spatialWeight[iy+radius][ix+radius] = 0.0;
			else spatialWeight[iy+radius][ix+radius] = exp(-float(iy*iy+ix*ix)/(sigmaSpatial*sigmaSpatial));
		}
	}

	for (iy=0; iy<COLOR_DIFF_SIZE; ++iy)
	{
		colorDiffWeight[iy] = exp(-float(iy)/(sigmaColor*sigmaColor));
		//(colorDiffWeight[iy]<1e-5)? colorDiffWeight[iy] = 0.0: NULL;
	}

	refinedFlow.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int dy, dx;
			cv::Vec3i anchorPix = weightColorImg[iy][ix];
			float normWeight = 0.0;
			cv::Vec2f weightedFlow(0.0, 0.0);
			for (dy=-radius; dy<=radius; ++dy)
			{
				for (dx=-radius; dx<=radius; ++dx)
				{
					int my, mx;
					my = iy+dy;
					mx = ix+dx;
					if (my<0 || my>=height || mx<0 || mx>=width) continue;
					cv::Vec3i tmpPix = weightColorImg[my][mx];
					int tmpColorDiff = (anchorPix[0]-tmpPix[0])*(anchorPix[0]-tmpPix[0])
						+ (anchorPix[1]-tmpPix[1])*(anchorPix[1]-tmpPix[1])
						+ (anchorPix[2]-tmpPix[2])*(anchorPix[2]-tmpPix[2]);
					(tmpColorDiff >= COLOR_DIFF_SIZE)? tmpColorDiff = COLOR_DIFF_SIZE-1: NULL;
					float totalWeight = spatialWeight[dy+radius][dx+radius] * colorDiffWeight[tmpColorDiff];

					weightedFlow += flowVec[my][mx]*totalWeight;
					normWeight += totalWeight;		

				}
			}
			refinedFlow[iy][ix][0] = weightedFlow[0]/normWeight;
			refinedFlow[iy][ix][1] = weightedFlow[1]/normWeight;
		}
	}
}

double SuperPatchmatch::CalcFlowStandardDeviation( const cv::Mat_<cv::Vec2f> &flowRes, const cv::Mat_<cv::Vec2f> &gtFlow )
{
	int height, width, iy, ix;
	height = flowRes.rows;
	width = flowRes.cols;
	double sumError = 0.0;
	double sumSquareError = 0.0;
	double validCnt = 0.0;

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// modify to account for scale change
			if (gtFlow[iy][ix][0]>1e5 || gtFlow[iy][ix][1]>1e5)
			{
				continue;
			}

			cv::Vec2f tmpGt = gtFlow[iy][ix];
			cv::Vec2f tmpR = flowRes[iy][ix];
			double tmpError = sqrt((tmpGt[0]-tmpR[0])*(tmpGt[0]-tmpR[0])+(tmpGt[1]-tmpR[1])*(tmpGt[1]-tmpR[1]));

			validCnt += 1.0;
			sumError += tmpError;
			sumSquareError += tmpError*tmpError;
		}
	}
	return sqrt(sumSquareError/validCnt-(sumError/validCnt)*(sumError/validCnt));
}


double SuperPatchmatch::CalcDiceCoefficient( const cv::Mat_<cv::Vec3b> &maskIn, const cv::Mat_<cv::Vec3b> &maskRef )
{
	int iy, ix;
	double sumInter = 0.0;
	double sumUnion = 0.0;
	int height = maskRef.rows;
	int width = maskRef.cols;

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			if (maskIn[iy][ix] == cv::Vec3b(0, 0, 0))
			{
				if (maskRef[iy][ix] == cv::Vec3b(0, 0, 0))
				{
					continue;
				}
				else sumUnion += 1.0;
			}
			else 
			{
				sumUnion += 1.0;
				if (maskRef[iy][ix] == maskIn[iy][ix])
				{
					sumInter += 1.0;
				}
			}
		}
	}

	// return sumInter/sumUnion;
	// [added] - 10-03-2014
	// the dice coef could be:
	return (sumInter*2)/(sumUnion+sumInter);

}


int SuperPatchmatch::SetDefaultParameters()
{
	// filter kernel related, for GF and CLMF
	g_filterKernelSize = 9;
	g_filterKernelBoundarySize = 2*g_filterKernelSize;
	g_filterKernelColorTau = 25;
	g_filterKernelEpsl = 100;

	// superpixel related
	g_spMethod = 0;
	g_spNumber = 300;
	g_spSize = 300;
	g_spSizeOrNumber = 1;


	g_iterTimes = 40;
	g_flowPrecision = 1;

	iterationTimes = g_iterTimes;
	upPrecision = g_flowPrecision;

	DO_LEFT = true;
	DO_RIGHT = false;
	useMaskTransfer = false;
	hasGtFlow = false;
	refMaxMotion = -1.0;

	return 0;
}

int SuperPatchmatch::CreateAndOrganizeSuperpixels()
{
	cv::Mat img1 = imLeftOrigin;
	cv::Mat img2 = imRightOrigin;

	cv::Mat_<int> labelLeft, labelRight;
	int numLabelLeft, numLabelRight;
	cv::Mat_<cv::Vec4i> subLeft, subRight;
	cv::Mat_<cv::Vec4i> spLeft, spRight;

	CalcTime ct;
	ct.Start();
	if (g_spMethod == 0)
	{
		numLabelLeft = CreateSLICSegments(img1, labelLeft, g_spNumber, g_spSize, g_spSizeOrNumber);
		GetSubImageRangeFromSegments(labelLeft, numLabelLeft, g_filterKernelBoundarySize, subLeft, spLeft);
		numLabelRight = CreateSLICSegments(img2, labelRight, g_spNumber, g_spSize, g_spSizeOrNumber);
		GetSubImageRangeFromSegments(labelRight, numLabelRight, g_filterKernelBoundarySize, subRight, spRight);
	}


	// draw out the segmented image
	/*if (g_spMethod == 0 || g_spMethod == 1)
	{
	cv::Mat_<cv::Vec3b> resImg;
	DrawContoursAroundSegments(img1, labelLeft, resImg);
	cv::imshow(WINDOW_SEGMENT_CONTOUR, resImg);
	cv::Mat_<cv::Vec3b> resImgRight;
	DrawContoursAroundSegments(img2, labelRight, resImgRight);
	cv::imshow(WINDOW_SEGMENT_CONTOUR_RIGHT, resImgRight);
	WriteOutImageResult(resImg, WINDOW_SEGMENT_CONTOUR, 0);
	WriteOutImageResult(resImgRight, WINDOW_SEGMENT_CONTOUR_RIGHT, 0);
	}*/


	printf("==================================================\n");
	ct.End("Created segments and sub-images");

	subRangeLeft = subLeft.clone();
	spRangeLeft = spLeft.clone();

	numOfLabelsLeft = numLabelLeft;
	segLabelsLeft = labelLeft.clone();

	subRangeRight = subRight.clone();
	spRangeRight= spRight.clone();

	numOfLabelsRight = numLabelRight;
	segLabelsRight = labelRight.clone();

	/*
	double minL, maxL;
	cv::minMaxLoc(spm.segLabelsLeft, &minL, &maxL);
	printf("Left: num = %d minL = %f maxL = %f\n", spm.numOfLabelsLeft, minL, maxL);
	cv::minMaxLoc(spm.segLabelsRight, &minL, &maxL);
	printf("Right: num = %d minL = %f maxL = %f\n", spm.numOfLabelsRight, minL, maxL);
	*/

	GetSuperpixelsListFromSegment(segLabelsLeft, numOfLabelsLeft, superpixelsListLeft);
	GetSuperpixelsListFromSegment(segLabelsRight, numOfLabelsRight, superpixelsListRight);
	printf("Got super-pixels list\n==================================================\n");

	return 0;
}
