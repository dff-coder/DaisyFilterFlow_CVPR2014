#include "BoxFilter_Header.h"

void CostBoxFilter::DoCostBoxFilter( const cv::Mat_<float> &costIn, int radius, cv::Mat_<float> &filteredCost )
{
	int height, width;
	height = costIn.rows;
	width = costIn.cols;

	// window normalize size
	normWin.create(height, width);
	normWin.setTo(1.0);
	TheBoxFilter(normWin, normWin, radius);
	
	cv::Mat_<float> p = costIn;

	cv::Mat_<float> mP;
	// the mean of cost p
	TheBoxFilter(p, mP, radius);
	//NewTheBoxFilter(p, mP, radius);
	mP /= normWin;

	filteredCost = mP.clone();
}

void CostBoxFilter::TheBoxFilter( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius )
{
	int height, width, iy, ix;
	height = bIn.rows;
	width = bIn.cols;

	bOut.create(height, width);
	cv::Mat_<float> cumHorSum(height, width);
	float *horSum = new float [width];
	for (iy=0; iy<height; ++iy)
	{
		float *bPtr = (float *)(bIn.ptr(iy));
		float *hPtr = horSum;
		float s = 0.0f;
		for (ix=0; ix<width; ++ix)
		{
			//s += bIn[iy][ix];
			//horSum[ix] = s;
			s += *bPtr++;
			*hPtr++ = s;
		}

		float *ptrR = horSum+radius;
		float *dPtr = (float *)(cumHorSum.ptr(iy));
		for (ix=0; ix<=radius; ++ix)
			*dPtr++ = *ptrR++;
		// cumHorSum[iy][ix] = horSum[ix+radius];


		float *ptrL = horSum;
		for (; ix<width-radius; ++ix)
			*dPtr++ = *ptrR++-*ptrL++;
		//cumHorSum[iy][ix] = horSum[ix+radius]-horSum[ix-radius-1];

		--ptrR;
		for (; ix<width; ++ix)
			*dPtr++ = *ptrR-*ptrL++;
		//cumHorSum[iy][ix] = horSum[width-1]-horSum[ix-radius-1];
	}

	const int W_FAC = width;
	float *colSum = new float [height];
	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0f;
		float *cuPtr = (float *)(cumHorSum.ptr(0))+ix;
		float *coPtr = colSum;
		for (iy=0; iy<height; ++iy, cuPtr+=W_FAC)
		{
			//s += cumHorSum[iy][ix];
			//colSum[iy] = s;
			s += *cuPtr;
			*coPtr++ = s;
		}

		float *ptrD = colSum+radius;
		float *dPtr = (float *)(bOut.ptr(0))+ix;

		for (iy=0; iy<=radius; ++iy, dPtr+=W_FAC)
			//bOut[iy][ix] = colSum[iy+radius];
			*dPtr = *ptrD++;

		float *ptrU = colSum;
		for (; iy<height-radius; ++iy, dPtr+=W_FAC)
			//bOut[iy][ix] = colSum[iy+radius]-colSum[iy-radius-1];
			*dPtr = *ptrD++-*ptrU++;

		--ptrD;
		for (; iy<height; ++iy, dPtr+=W_FAC)
			// bOut[iy][ix] = colSum[height-1]-colSum[iy-radius-1];
			*dPtr = *ptrD-*ptrU++;
	}

	delete [] horSum;
	delete [] colSum;
}


void CostBoxFilter::TheBoxFilterArrayForm( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius )
{
	int height, width, iy, ix;
	height = bIn.rows;
	width = bIn.cols;

	bOut.create(height, width);
	cv::Mat_<float> cumHorSum(height, width);
	float *horSum = new float [width];
	for (iy=0; iy<height; ++iy)
	{
		float s = 0.0f;
		for (ix=0; ix<width; ++ix)
		{
			s += bIn[iy][ix];
			horSum[ix] = s;
		}

		for (ix=0; ix<=radius; ++ix)
			cumHorSum[iy][ix] = horSum[ix+radius];

		for (; ix<width-radius; ++ix)
			cumHorSum[iy][ix] = horSum[ix+radius]-horSum[ix-radius-1];

		for (; ix<width; ++ix)
			cumHorSum[iy][ix] = horSum[width-1]-horSum[ix-radius-1];
	}

	float *colSum = new float [height];
	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0f;
		for (iy=0; iy<height; ++iy)
		{
			s += cumHorSum[iy][ix];
			colSum[iy] = s;
		}

		for (iy=0; iy<=radius; ++iy)
			bOut[iy][ix] = colSum[iy+radius];

		for (; iy<height-radius; ++iy)
			bOut[iy][ix] = colSum[iy+radius]-colSum[iy-radius-1];

		for (; iy<height; ++iy)
			bOut[iy][ix] = colSum[height-1]-colSum[iy-radius-1];
	}

	delete [] horSum;
	delete [] colSum;
}