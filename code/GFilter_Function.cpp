#include "GFilter_Header.h"

void GFilter::TheBoxFilterArrayForm( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius )
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

void GFilter::InitiateGuidance( const cv::Mat_<cv::Vec3b> &gImg, int radius, float epsl )
{
	cv::Mat_<float> varIBB, varIBG, varIBR, varIGG, varIGR, varIRR;
	cv::Mat_<float> mIBB, mIBG, mIBR, mIGG, mIGR, mIRR;

	int height, width;
	height = gImg.rows;
	width = gImg.cols;

	cv::Mat gArr[3];
	cv::split(gImg, gArr);
	gArr[0].convertTo(iB, CV_32FC1, 1/255.0f);
	gArr[1].convertTo(iG, CV_32FC1, 1/255.0f);
	gArr[2].convertTo(iR, CV_32FC1, 1/255.0f);

	// window normalize size
	normWin.create(height, width);
	normWin.setTo(1.0);
	TheBoxFilter(normWin, normWin, radius);

	// mean of guidance image I
	TheBoxFilter(iB, mIB, radius);
	TheBoxFilter(iG, mIG, radius);
	TheBoxFilter(iR, mIR, radius);

	mIB /= normWin;
	mIG /= normWin;
	mIR /= normWin;

	// variance of guidance image I
	TheBoxFilter(iB.mul(iB), mIBB, radius);
	TheBoxFilter(iB.mul(iG), mIBG, radius);
	TheBoxFilter(iB.mul(iR), mIBR, radius);
	TheBoxFilter(iG.mul(iG), mIGG, radius);
	TheBoxFilter(iG.mul(iR), mIGR, radius);
	TheBoxFilter(iR.mul(iR), mIRR, radius);

	mIBB /= normWin;
	mIBG /= normWin;
	mIBR /= normWin;
	mIGG /= normWin;
	mIGR /= normWin;
	mIRR /= normWin;

	varIBB = mIBB - mIB.mul(mIB);
	varIBG = mIBG - mIB.mul(mIG);
	varIBR = mIBR - mIB.mul(mIR);
	varIGG = mIGG - mIG.mul(mIG);
	varIGR = mIGR - mIG.mul(mIR);
	varIRR = mIRR - mIR.mul(mIR);

	// calculate inverse
	int iy, ix;
	//invIU.create(height, width);			
	invIU.resize(height);
	cv::Matx33f epsU(
		epsl, 0.0, 0.0,
		0.0, epsl, 0.0,
		0.0, 0.0, epsl);
	for (iy=0; iy<height; ++iy)
	{
		invIU[iy].resize(width);
		for (ix=0; ix<width; ++ix)
		{
			//printf("0: %d %d\n", iy, ix);
			cv::Matx33f tmpM(
				varIBB[iy][ix], varIBG[iy][ix], varIBR[iy][ix],
				varIBG[iy][ix], varIGG[iy][ix], varIGR[iy][ix],
				varIBR[iy][ix], varIGR[iy][ix], varIRR[iy][ix]);
				

			invIU[iy][ix] = (tmpM+epsU).inv();
			//invIU(iy, ix) = epsU.inv();
			//printf("%d %d %d %d\n", iy, ix, invIU.rows, invIU.cols);
			//invIU[iy][ix] = epsU.inv();
		}
	}
}


void GFilter::InitiateGuidance( const cv::Mat_<cv::Vec3f> &gImg, int radius, float epsl )
{
	cv::Mat_<float> varIBB, varIBG, varIBR, varIGG, varIGR, varIRR;
	cv::Mat_<float> mIBB, mIBG, mIBR, mIGG, mIGR, mIRR;

	int height, width;
	height = gImg.rows;
	width = gImg.cols;

	cv::Mat gArr[3];
	cv::split(gImg, gArr);
	gArr[0].convertTo(iB, CV_32FC1, 1/255.0f);
	gArr[1].convertTo(iG, CV_32FC1, 1/255.0f);
	gArr[2].convertTo(iR, CV_32FC1, 1/255.0f);

	// window normalize size
	normWin.create(height, width);
	normWin.setTo(1.0);
	TheBoxFilter(normWin, normWin, radius);

	// mean of guidance image I
	TheBoxFilter(iB, mIB, radius);
	TheBoxFilter(iG, mIG, radius);
	TheBoxFilter(iR, mIR, radius);

	mIB /= normWin;
	mIG /= normWin;
	mIR /= normWin;

	// variance of guidance image I
	TheBoxFilter(iB.mul(iB), mIBB, radius);
	TheBoxFilter(iB.mul(iG), mIBG, radius);
	TheBoxFilter(iB.mul(iR), mIBR, radius);
	TheBoxFilter(iG.mul(iG), mIGG, radius);
	TheBoxFilter(iG.mul(iR), mIGR, radius);
	TheBoxFilter(iR.mul(iR), mIRR, radius);

	mIBB /= normWin;
	mIBG /= normWin;
	mIBR /= normWin;
	mIGG /= normWin;
	mIGR /= normWin;
	mIRR /= normWin;

	varIBB = mIBB - mIB.mul(mIB);
	varIBG = mIBG - mIB.mul(mIG);
	varIBR = mIBR - mIB.mul(mIR);
	varIGG = mIGG - mIG.mul(mIG);
	varIGR = mIGR - mIG.mul(mIR);
	varIRR = mIRR - mIR.mul(mIR);

	// calculate inverse
	int iy, ix;
	//invIU.create(height, width);			
	invIU.resize(height);
	cv::Matx33f epsU(
		epsl, 0.0, 0.0,
		0.0, epsl, 0.0,
		0.0, 0.0, epsl);
	for (iy=0; iy<height; ++iy)
	{
		invIU[iy].resize(width);
		for (ix=0; ix<width; ++ix)
		{
			//printf("0: %d %d\n", iy, ix);
			cv::Matx33f tmpM(
				varIBB[iy][ix], varIBG[iy][ix], varIBR[iy][ix],
				varIBG[iy][ix], varIGG[iy][ix], varIGR[iy][ix],
				varIBR[iy][ix], varIGR[iy][ix], varIRR[iy][ix]);

			// [added - 2013-10-18], try to avoid singular matrix
			// but no effect
			tmpM = tmpM+epsU;
			/*double dtrm = cv::determinant(tmpM);
			if (dtrm <= std::numeric_limits<double>::epsilon())
			{
				tmpM(0, 0) += 0.01;
				tmpM(1, 1) += 0.01;
				tmpM(2, 2) += 0.01;
			}*/

			invIU[iy][ix] = tmpM.inv();
			//invIU(iy, ix) = epsU.inv();
			//printf("%d %d %d %d\n", iy, ix, invIU.rows, invIU.cols);
			//invIU[iy][ix] = epsU.inv();
		}
	}
}


/*
[hei, wid] = size(p);
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.

	mean_I_r = boxfilter(I(:, :, 1), r) ./ N;
mean_I_g = boxfilter(I(:, :, 2), r) ./ N;
mean_I_b = boxfilter(I(:, :, 3), r) ./ N;

mean_p = boxfilter(p, r) ./ N;

mean_Ip_r = boxfilter(I(:, :, 1).*p, r) ./ N;
mean_Ip_g = boxfilter(I(:, :, 2).*p, r) ./ N;
mean_Ip_b = boxfilter(I(:, :, 3).*p, r) ./ N;

% covariance of (I, p) in each local patch.
	cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;

% variance of I in each local patch: the matrix Sigma in Eqn (14).
	% Note the variance in each local patch is a 3x3 symmetric matrix:
%           rr, rg, rb
	%   Sigma = rg, gg, gb
	%           rb, gb, bb
	var_I_rr = boxfilter(I(:, :, 1).*I(:, :, 1), r) ./ N - mean_I_r .*  mean_I_r; 
var_I_rg = boxfilter(I(:, :, 1).*I(:, :, 2), r) ./ N - mean_I_r .*  mean_I_g; 
var_I_rb = boxfilter(I(:, :, 1).*I(:, :, 3), r) ./ N - mean_I_r .*  mean_I_b; 
var_I_gg = boxfilter(I(:, :, 2).*I(:, :, 2), r) ./ N - mean_I_g .*  mean_I_g; 
var_I_gb = boxfilter(I(:, :, 2).*I(:, :, 3), r) ./ N - mean_I_g .*  mean_I_b; 
var_I_bb = boxfilter(I(:, :, 3).*I(:, :, 3), r) ./ N - mean_I_b .*  mean_I_b; 

a = zeros(hei, wid, 3);
for y=1:hei
	for x=1:wid        
		Sigma = [var_I_rr(y, x), var_I_rg(y, x), var_I_rb(y, x);
var_I_rg(y, x), var_I_gg(y, x), var_I_gb(y, x);
var_I_rb(y, x), var_I_gb(y, x), var_I_bb(y, x)];
Sigma = Sigma + eps * eye(3);

cov_Ip = [cov_Ip_r(y, x), cov_Ip_g(y, x), cov_Ip_b(y, x)];        

a(y, x, :) = cov_Ip * inv(Sigma + eps * eye(3)); % Eqn. (14) in the paper;
end
	end

	b = mean_p - a(:, :, 1) .* mean_I_r - a(:, :, 2) .* mean_I_g - a(:, :, 3) .* mean_I_b; % Eqn. (15) in the paper;

q = (boxfilter(a(:, :, 1), r).* I(:, :, 1)...
	+ boxfilter(a(:, :, 2), r).* I(:, :, 2)...
	+ boxfilter(a(:, :, 3), r).* I(:, :, 3)...
	+ boxfilter(b, r)) ./ N;  % Eqn. (16) in the paper;
end*/

void GFilter::DoGuidedFilter( const cv::Mat_<float> &costIn, int radius, float epsl, cv::Mat_<float> &filteredCost )
{
	cv::Mat_<float> mIRp, mIGp, mIBp;
	cv::Mat_<float> covIRp, covIGp, covIBp;
	cv::Mat_<float> mP;

	cv::Mat_<float> aPR, aPG, aPB;
	cv::Mat_<float> bP;

	cv::Mat_<float> maPR, maPG, maPB;
	cv::Mat_<float> mbP;

	cv::Mat_<float> p = costIn;
	int height, width;
	height = p.rows;
	width = p.cols;

	// the mean of cost p
	TheBoxFilter(p, mP, radius);
	mP /= normWin;

	// the convariance of I and p
	TheBoxFilter(p.mul(iB), mIBp, radius);
	TheBoxFilter(p.mul(iG), mIGp, radius);
	TheBoxFilter(p.mul(iR), mIRp, radius);
	covIBp = mIBp/normWin-mP.mul(mIB);
	covIGp = mIGp/normWin-mP.mul(mIG);
	covIRp = mIRp/normWin-mP.mul(mIR);

	int iy, ix;
	aPB.create(height, width);
	aPG.create(height, width);
	aPR.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			//cout << cv::Mat(invIU[iy][ix]) << endl;

			//cout << invIU[iy][ix](0, 0) << invIU[iy][ix](0, 1) << invIU[iy][ix](0, 2) << endl;

			aPB[iy][ix] = invIU[iy][ix](0, 0)*covIBp[iy][ix] 
			+ invIU[iy][ix](0, 1)*covIGp[iy][ix] 
			+ invIU[iy][ix](0, 2)*covIRp[iy][ix];

			aPG[iy][ix] = invIU[iy][ix](1, 0)*covIBp[iy][ix] 
			+ invIU[iy][ix](1, 1)*covIGp[iy][ix] 
			+ invIU[iy][ix](1, 2)*covIRp[iy][ix];

			aPR[iy][ix] = invIU[iy][ix](2, 0)*covIBp[iy][ix] 
			+ invIU[iy][ix](2, 1)*covIGp[iy][ix] 
			+ invIU[iy][ix](2, 2)*covIRp[iy][ix];
		}
	}

	bP = mP - (aPB.mul(mIB) + aPG.mul(mIG) + aPR.mul(mIR));

	// another average
	TheBoxFilter(aPB, maPB, radius);
	TheBoxFilter(aPG, maPG, radius);
	TheBoxFilter(aPR, maPR, radius);
	TheBoxFilter(bP, mbP, radius);

	maPB /= normWin;
	maPG /= normWin;
	maPR /= normWin;
	mbP /= normWin;

	filteredCost = maPB.mul(iB) + maPG.mul(iG) + maPR.mul(iR) + mbP;
}



void GFilter::ClearUp()
{
	int iy, ix;
	for (iy=0; iy<invIU.size(); ++iy)
	{
		for (ix=0; ix<invIU[iy].size(); ++ix)
		{
			cv::Mat(invIU[iy][ix]).release();
		}
		invIU[iy].clear();
	}
	invIU.clear();

//	varIBB.release(); varIBG.release(); varIBR.release(); varIGG.release(); varIGR.release(); varIRR.release();
//	mIBB.release(); mIBG.release(); mIBR.release(); mIGG.release(); mIGR.release(); mIRR.release();
	mIR.release(); mIG.release(); mIB.release();
	//mIRp.release(); mIGp.release(); mIBp.release();
	//covIRp.release(); covIGp.release(); covIBp.release();
	//mP.release();
	normWin.release();

	//aPR.release(); aPG.release(); aPB.release();
	//bP.release();

	//maPR.release(); maPG.release(); maPB.release();
	//mbP.release();

	iR.release(); iG.release(); iB.release();

	//guidedI.release();
	//costP.release();
}


GFilter::~GFilter()
{
	ClearUp();
}


void GFilter::NewDoGuidedFilter( const cv::Mat_<float> &costIn, int radius, float epsl, cv::Mat_<float> &filteredCost )
{
	cv::Mat_<float> mIRp, mIGp, mIBp;
	cv::Mat_<float> covIRp, covIGp, covIBp;
	cv::Mat_<float> mP;

	cv::Mat_<float> aPR, aPG, aPB;
	cv::Mat_<float> bP;

	cv::Mat_<float> maPR, maPG, maPB;
	cv::Mat_<float> mbP;

	cv::Mat_<float> p = costIn;
	int height, width;
	height = p.rows;
	width = p.cols;

	// the mean of cost p
	TheBoxFilter(p, mP, radius);
	mP /= normWin;

	// the convariance of I and p
	TheBoxFilter(p.mul(iB), mIBp, radius);
	TheBoxFilter(p.mul(iG), mIGp, radius);
	TheBoxFilter(p.mul(iR), mIRp, radius);

	covIBp = mIBp/normWin-mP.mul(mIB);
	covIGp = mIGp/normWin-mP.mul(mIG);
	covIRp = mIRp/normWin-mP.mul(mIR);

	int iy, ix;
	aPB.create(height, width);
	aPG.create(height, width);
	aPR.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Matx31f tmpCovIp(covIBp[iy][ix], covIGp[iy][ix], covIRp[iy][ix]);
			cv::Matx31f tmpAP = invIU[iy][ix]*tmpCovIp;
			aPB[iy][ix] = tmpAP.val[0];
			aPG[iy][ix] = tmpAP.val[1];
			aPR[iy][ix] = tmpAP.val[2];
		}
	}

	bP = mP - (aPB.mul(mIB) + aPG.mul(mIG) + aPR.mul(mIR));

	// another average
	TheBoxFilter(aPB, maPB, radius);
	TheBoxFilter(aPG, maPG, radius);
	TheBoxFilter(aPR, maPR, radius);
	TheBoxFilter(bP, mbP, radius);

	maPB /= normWin;
	maPG /= normWin;
	maPR /= normWin;
	mbP /= normWin;

	filteredCost = maPB.mul(iB) + maPG.mul(iG) + maPR.mul(iR) + mbP;
}


void GFilter::TheBoxFilter( const cv::Mat_<float> &bIn, cv::Mat_<float> &bOut, int radius )
{
	int height, width, iy, ix;
	height = bIn.rows;
	width = bIn.cols;

	bOut.create(height, width);
	cv::Mat_<float> cumHorSum(height, width);
	float *horSum = new float [width];

// #pragma omp parallel for private(iy, ix) schedule(guided, 1)
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
