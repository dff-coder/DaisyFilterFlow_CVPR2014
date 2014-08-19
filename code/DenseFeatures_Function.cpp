#include "opencv2/opencv.hpp"
#include "DenseFeatures_Header.h"
#include "Program_Control_Header.h"

#include "daisy/daisy.h"
using namespace kutility;

// for dense daisy
int CreateDenseDaisy(const cv::Mat &imgIn, float scale, float angle, cv::Mat &descrpOut)
{
	cv::Mat imgGray = imgIn;

	// setting of daisy library
	int verbose_level = 0;

	//int rad   = 15;
	int rad = scale;
	int radq  =  DAISY_FEATURE_LAYERS;
	int thq   =  8;
	int histq =  8;

	int nrm_type = NRM_PARTIAL;
	//int nrm_type = NRM_SIFT;

	//bool disable_interpolation = false;
	bool disable_interpolation = true;

	// associate pointer
	uchar *im = imgGray.data;
	int h = imgGray.rows;
	int w = imgGray.cols;

	daisy* desc = new daisy();

	if( disable_interpolation ) desc->disable_interpolation();

	desc->set_image(im,h,w);
	//	deallocate(im);
	desc->verbose( verbose_level );
	desc->set_parameters(rad, radq, thq, histq);
	if( nrm_type == 0 ) desc->set_normalization( NRM_PARTIAL );
	if( nrm_type == 1 ) desc->set_normalization( NRM_FULL );
	if( nrm_type == 2 ) desc->set_normalization( NRM_SIFT );

	// !! this part is optional. You don't need to set the workspace memory
	//int ws = desc->compute_workspace_memory();
	//float* workspace = new float[ ws ];
	//desc->set_workspace_memory( workspace, ws);
	// !! this part is optional. You don't need to set the workspace memory

	desc->initialize_single_descriptor_mode();

	// !! this part is optional. You don't need to set the descriptor memory
	// int ds = desc->compute_descriptor_memory();
	// float* descriptor_mem = new float[ds];
	// desc->set_descriptor_memory( descriptor_mem, ds );
	// !! this part is optional. You don't need to set the descriptor memory
	//printf("here 1-1-1\n");
	desc->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)
	// the descriptors are not normalized yet
	desc->normalize_descriptors();
	//printf("here 1-1-2\n");
	// save out the dense daisy descriptors
	int iy, ix, descSize;
	descSize = desc->descriptor_size();
	descrpOut.create(h*w, descSize, CV_32FC1);
	for (iy=0; iy<h; ++iy)
	{
		for (ix=0; ix<w; ++ix)
		{
			float* thor = NULL;
			desc->get_descriptor(iy, ix, thor);
			memcpy(descrpOut.ptr(iy*w+ix), thor, descSize*sizeof(float));
		}
	}
	//printf("here 1-1-3\n");
	delete desc;
	//printf("here 1-1-4\n");
	return 1;
}

daisy *InitializeOneDaisyDesc(const cv::Mat &imgIn, float scale, int descLayers)
{
	cv::Mat imgGray = imgIn;

	// setting of daisy library
	int verbose_level = 1;

	//int rad   = 15;
	int rad = scale;
	//int radq  =  2;
	int radq = descLayers;
	int thq   =  8;
	int histq =  8;

	int nrm_type = NRM_PARTIAL;
	//int nrm_type = NRM_SIFT;

	bool disable_interpolation = true;

	// associate pointer
	uchar *im = imgGray.data;
	int h = imgGray.rows;
	int w = imgGray.cols;

	daisy* desc = new daisy();

	if( disable_interpolation ) desc->disable_interpolation();

	desc->set_image(im,h,w);
	//	deallocate(im);
	desc->verbose( verbose_level );
	desc->set_parameters(rad, radq, thq, histq);
	if( nrm_type == 0 ) desc->set_normalization( NRM_PARTIAL );
	if( nrm_type == 1 ) desc->set_normalization( NRM_FULL );
	if( nrm_type == 2 ) desc->set_normalization( NRM_SIFT );

	// !! this part is optional. You don't need to set the workspace memory
	//int ws = desc->compute_workspace_memory();
	//float* workspace = new float[ ws ];
	//desc->set_workspace_memory( workspace, ws);
	// !! this part is optional. You don't need to set the workspace memory

	desc->initialize_single_descriptor_mode();

	// !! this part is optional. You don't need to set the descriptor memory
	// int ds = desc->compute_descriptor_memory();
	// float* descriptor_mem = new float[ds];
	// desc->set_descriptor_memory( descriptor_mem, ds );
	// !! this part is optional. You don't need to set the descriptor memory

	// [added] - 2013-03-30, to store precomputed normalized histogram
	desc->PrecomputeNormalizedHistogram();

	return desc;
}

int ExtractSubImageDaisyDescriptors(daisy *desc, cv::Mat_<float> &descOut, float py, float px, float step, int ori, int h, int w, float limH, float limW)
{
	//printf("here 0 %d %d py = %f px = %f\n", w, h, py, px);
	descOut.create(w*h, DAISY_FEATURE_LENGTH);
	//printf("here 0-1\n");
	descOut.setTo(cv::Scalar(0.0));
	//printf("here 0-2\n");
	int iy, ix;
	float cy, cx, ssinOri, scosOri;
	ssinOri = step*sin((float)ori/180.0*M_PI);
	scosOri = step*cos((float)ori/180.0*M_PI);
	float ry, rx; // the first pos of each row
	ry = py; rx = px;
	//printf("here 1\n");
	//float *thor = new float [DAISY_FEATURE_LENGTH];
	for (iy=0; iy<h; ++iy)
	{
		cy = ry; cx = rx;
		float *ptr = (float *)(descOut.ptr(iy*w));
		//printf("here 2 %f %f %f %f\n", cy, cx, ssinOri, scosOri);
		
		for (ix=0; ix<w; ++ix)
		{			
			//memset(thor, 0, sizeof(float)*desc->descriptor_size());
			//desc->get_descriptor(std::min<double>(limH-1.0, std::max<double>(cy, 0.0)), std::min<double>(limW-1.0, std::max<double>(cx, 0.0)), ori, ptr);
			desc->ExtractDescriptorFromNormalizedHistogram(ptr, std::min<double>(limH-1.0, std::max<double>(cy, 0.0)), std::min<double>(limW-1.0, std::max<double>(cx, 0.0)), ori);
			ptr += DAISY_FEATURE_LENGTH;
			//printf("here 2-1\n");
			//memcpy(ptr, thor, sizeof(float)*DAISY_FEATURE_LENGTH);
			//printf("here 2-2\n");
			//printf("here 2-3\n");
			cy = cy+ssinOri;
			cx = cx+scosOri;
		}
		ry = ry+scosOri;
		rx = rx-ssinOri;
		//printf("here 2\n");
	}
	//delete [] thor;
	//printf("here 3\n");
	return 1;
}

int ExtractAndComputeSubImageDaisyDescriptorsCost( daisy *desc, cv::Mat_<float> &costOut, cv::Mat_<float> &descRef, float py, float px, float step, int ori, int h, int w, cv::Vec4f &tmpFl )
{
	//printf("here 0 %d %d py = %f px = %f\n", w, h, py, px);
	//descOut.create(w*h, DAISY_FEATURE_LENGTH);
	//printf("here 0-1\n");
	//descOut.setTo(cv::Scalar(0.0));
	//printf("here 0-2\n");
	int iy, ix;
	float cy, cx, ssinOri, scosOri;
	ssinOri = step*sin((float)ori/180.0*M_PI);
	scosOri = step*cos((float)ori/180.0*M_PI);
	float ry, rx; // the first pos of each row
	ry = py; rx = px;
	//printf("here 1\n");
	//float *thor = new float [DAISY_FEATURE_LENGTH];
	cv::Matx23f tranMat(scosOri, -ssinOri, tmpFl[0], 
		ssinOri, scosOri, tmpFl[1]);

	for (iy=0; iy<h; ++iy)
	{
		cy = ry; cx = rx;
		float *ptr = (float *)(descRef.ptr(iy*w));
		//printf("here 2 %f %f %f %f\n", cy, cx, ssinOri, scosOri);

		for (ix=0; ix<w; ++ix)
		{			
			//memset(thor, 0, sizeof(float)*desc->descriptor_size());
			// desc->get_descriptor(std::min<double>(limH-1.0, std::max<double>(cy, 0.0)), std::min<double>(limW-1.0, std::max<double>(cx, 0.0)), ori, ptr);
			costOut[iy][ix] = desc->ExtractAndCalculateTheDistanceFromNormalizedHistogram(ptr, cy, cx, ori);
			
			//cv::Matx31f p0(px-tmpFl[0]+ix, py-tmpFl[1]+iy, 1.0);
			//cv::Matx21f p1 = tranMat*p0;
			//printf("iy = %d ix = %d, cy = %f cx = %f, ori = %d, p1 = %f %f, ptr = %p\n", iy, ix, cy, cx, ori, p1.val[1], p1.val[0], ptr);
			//printf("py = %f px = %f, fl = %f %f, diff = %f %f\n", py, px, tmpFl[0], tmpFl[1], p1.val[0]-cx, p1.val[1]-cy);
			//costOut[iy][ix] = desc->ExtractAndCalculateTheDistanceFromNormalizedHistogram(ptr, p1.val[1], p1.val[0], ori);
			ptr += DAISY_FEATURE_LENGTH;
			//printf("here 2-1\n");
			//memcpy(ptr, thor, sizeof(float)*DAISY_FEATURE_LENGTH);
			//printf("here 2-2\n");
			//printf("here 2-3\n");
			cy = cy+ssinOri;
			cx = cx+scosOri;
		}
		ry = ry+scosOri;
		rx = rx-ssinOri;
		//printf("here 2\n");
	}
	//delete [] thor;
	//printf("here 3\n");
	return 1;
}


int NewlyExtractAndComputeSubImageDaisyDescriptorsCost( daisy *desc, cv::Mat_<float> &costOut, const cv::Mat_<float> &descRef, float oy, float ox, int h, int w, float dy, float dx, float step, int ori)
{
	float ssinOri, scosOri;
	ssinOri = step*sin((float)ori/180.0*M_PI);
	scosOri = step*cos((float)ori/180.0*M_PI);

	//printf("%f %f %f y = %f x = %f fl = %f %f\n", step, ssinOri, scosOri, oy, ox, dy, dx);
	cv::Matx23f tranMat(scosOri, -ssinOri, dx, ssinOri, scosOri, dy);
	int cy, cx;
	int py, px;
	py = oy; 
	for (cy=0; cy<h; ++cy, ++py)
	{
		float *ptr = (float *)(descRef.ptr(cy*w));
		px = ox;
		for (cx=0; cx<w; ++cx, ++px)
		{
			cv::Matx31f p0(px, py, 1.0);
			cv::Matx21f p1 = tranMat*p0;
			//cv::Matx21f p1(px+dx, py+dy);
			// printf("%f %f dx = %f dy = %f px = %d py = %d oy = %f ox = %f\n", p1.val[0], p1.val[1], dx, dy, px, py, oy, ox);
			costOut[cy][cx] = desc->ExtractAndCalculateTheDistanceFromNormalizedHistogram(ptr, p1.val[1], p1.val[0], ori);
			ptr += DAISY_FEATURE_LENGTH;
		}
	}
	return 1;
}