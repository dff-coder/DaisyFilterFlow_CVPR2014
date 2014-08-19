#include "../code/General_Include_Header.h"

int CalcForwardFlowUsingDaisyFilterFlowMethod(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &forwardFlow, int option = 0) {
	SuperPatchmatch spm;

	// set default paramters
	spm.SetDefaultParameters();

	// input image pairs
	spm.imLeftOrigin = img1.clone();
	spm.imRightOrigin = img2.clone();

	// create superpixels
	spm.CreateAndOrganizeSuperpixels();

	// run daisy filter flow main method
	cv::Mat_<cv::Vec2f> flowResult;
	spm.RunDaisyFilterFlow(flowResult);

	// copy and save out result
	forwardFlow = flowResult.clone();
	spm.WriteFlowFile("Forward_Flow_Result.flo", flowResult, flowResult.rows, flowResult.cols);

	// show the warped dst image using src->dst flow
	// i.e. show look similar to the src image's structure
	cv::Mat_<cv::Vec3b> fwReSrcImg;
	spm.WarpingReverseAffine(spm.bestLeftDaisyFlow, spm.imRightOrigin, fwReSrcImg, spm.scaleSCoefRight);
	spm.WriteOutImageResult(fwReSrcImg, "Src_Img_Warpped_from_Dst_Img_By_Forward_Flow");
	return 0;
}

int main() {
	string f1, f2;
	// WinAPI to call GUi to load two images
	ScanFile::GUI_GetFileName(f1);
	ScanFile::GUI_GetFileName(f2);

	// read in two images
	cv::Mat i1, i2;
	i1 = cv::imread(f1);
	i2 = cv::imread(f2);

	// calculated forward flow, stored in conventional 
	cv::Mat flow;
	CalcForwardFlowUsingDaisyFilterFlowMethod(i1, i2, flow);
	return 0;
}
