//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                                License Agreement
//                       For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <precomp.hpp>
#include <opencv2/contrib/hd_color_model.hpp>
#include <vector>

using namespace std;

namespace cv {
namespace HT {

const int def_noOfBins[] = {16, 16, 16};
const bool def_useDepth = false;
float def_hranges[] = {0, 180};
float def_sranges[] = {0, 256};
float def_vranges[] = {0, 256};
float def_dranges[] = {400, 7000}; 	// approx. to be replacwith exact value

HDcolorModel::HDcolorModel(void) {
	useDepth = def_useDepth;
	for(int i=0; i<3; i++) {
		noOfBins[i] = def_noOfBins[i];
	}
	noOfBins[3] = 0;
	frameSize = Size(0, 0);
	histRange[0][0] = def_hranges[0];
	histRange[0][1] = def_hranges[1];
	histRange[1][0] = def_sranges[0];
	histRange[1][1] = def_sranges[1];
	histRange[2][0] = def_vranges[0];
	histRange[2][1] = def_vranges[1];
	histRange[3][0] = def_dranges[0];
	histRange[3][1] = def_dranges[1];
	paramInit = true;
	detectorInit = false;
}

HDcolorModel::HDcolorModel(vector<int> _noOfBins, bool _useDepth) {
	for(int i=0; i<3; i++) {
		noOfBins[i] = _noOfBins[i];
	}
	if(_useDepth == true)
		noOfBins[3] = _noOfBins[3];
	histRange[0][0] = def_hranges[0];
	histRange[0][1] = def_hranges[1];
	histRange[1][0] = def_sranges[0];
	histRange[1][1] = def_sranges[1];
	histRange[2][0] = def_vranges[0];
	histRange[2][1] = def_vranges[1];
	histRange[3][0] = def_dranges[0];
	histRange[3][1] = def_dranges[1];
	frameSize = Size(0, 0);
	useDepth = def_useDepth;
	paramInit = true;
	detectorInit = false;
}

// here the ordering of parameters in the vector
void HDcolorModel::getParams(vector<int> intParams, vector<double> doubleParams) const {
	intParams.clear();
	intParams.push_back(noOfBins[0]);
	intParams.push_back(noOfBins[1]);
	intParams.push_back(noOfBins[2]);
	intParams.push_back(noOfBins[3]);
	intParams.push_back(useDepth);
	intParams.push_back(frameSize.height);
	intParams.push_back(frameSize.width);

	doubleParams.clear();
	doubleParams.push_back(histRange[0][0]);
	doubleParams.push_back(histRange[0][1]);
	doubleParams.push_back(histRange[1][0]);
	doubleParams.push_back(histRange[1][1]);
	doubleParams.push_back(histRange[2][0]);
	doubleParams.push_back(histRange[2][1]);
	doubleParams.push_back(histRange[3][0]);
	doubleParams.push_back(histRange[3][1]);
}

void HDcolorModel::setParams(vector<int> intParams, vector<double> doubleParams) {
	CV_Assert(intParams.size() == intParamsN);
	CV_Assert(doubleParams.size() == doubleParamsN);

	noOfBins[0] = intParams[0];
	noOfBins[1] = intParams[1];
	noOfBins[2] = intParams[2];
	noOfBins[3] = intParams[3];

	useDepth = intParams[4];
	frameSize.height = intParams[5];
	frameSize.width = intParams[6];

	histRange[0][0] = doubleParams[0];
	histRange[0][1] = doubleParams[1];
	histRange[1][0] = doubleParams[2];
	histRange[1][1] = doubleParams[3];
	histRange[2][0] = doubleParams[4];
	histRange[2][1] = doubleParams[5];
	histRange[3][0] = doubleParams[6];
	histRange[3][1] = doubleParams[7];

	paramInit = true;
	detectorInit = false;
}

bool HDcolorModel::initialize(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool _useDepth) {
	CV_Assert(_rgbImg.type() == CV_8UC3 && _mask.type() == CV_8UC1);

	useDepth = _useDepth;

	if(useDepth) {
		CV_Assert(_depthImg.type() == CV_16UC1);
		CV_Assert(_rgbImg.size() == _depthImg.size());
	}

	if(paramInit) {
		createColorModel(_rgbImg, _depthImg, _mask);
	}

	return detectorInit;
}

void HDcolorModel::detect(Mat & _rgbImg, Mat & _depthImg, OutputArray _probImg) {
	if(detectorInit == false)
		return; // ??? do what

	Mat hsvImg = _rgbImg;
	//cvtColor(_rgbImg, hsvImg, CV_BGR2HSV);
	vector<Mat> channel;
	split(hsvImg, channel);

	const float* range[] = {histRange[0], histRange[1], histRange[2], histRange[3]};
	calcBackProject(&channel[0], 1, 0, hist[0], backPro[0], range);
	calcBackProject(&channel[1], 1, 0, hist[1], backPro[1], range+1);
	calcBackProject(&channel[2], 1, 0, hist[2], backPro[2], range+2);
	if(useDepth)
		calcBackProject(&_depthImg, 1, 0, hist[3], backPro[3], range+3);

	multiply(backPro[0], backPro[1], _probImg, 1./255.0);
	multiply(backPro[2], _probImg, _probImg, 1./255.0);
	if(useDepth)
		multiply(backPro[3], _probImg, _probImg, 1, CV_8UC1);
}

void HDcolorModel::createColorModel(Mat &_rgbImg, Mat & _depthImg, Mat & _mask) {
	vector<Mat> channel;
	Mat hsvImg = _rgbImg;
	//cvtColor(_rgbImg, hsvImg, CV_BGR2HSV);
	split(hsvImg, channel);
	const float* range[] = {histRange[0], histRange[1], histRange[2], histRange[3]};
	calcHist(&channel[0], 1, 0, _mask, hist[0], 1, &noOfBins[0], range);
	calcHist(&channel[1], 1, 0, _mask, hist[1], 1, &noOfBins[1], range+1);
	calcHist(&channel[2], 1, 0, _mask, hist[2], 1, &noOfBins[2], range+2);

	if(useDepth) {
		calcHist(&_depthImg, 1, 0, _mask, hist[3], 1, &noOfBins[2], range+3);
	}

	detectorInit = true;
}
}
}

