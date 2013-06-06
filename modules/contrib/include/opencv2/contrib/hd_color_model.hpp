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

#ifndef _HDCOLORMODEL_HPP_
#define _HDCOLORMODEL_HPP_

#include <opencv2/contrib/hand_detector.hpp>
#include "opencv2/core.hpp"
#include <vector>

using namespace std;

namespace cv {
namespace HT {

class CV_EXPORTS HDcolorModel: public HandDetector {
protected:

	static const uint intParamsN = 7;
	static const uint doubleParamsN = 8;

	// specifies if the depth is to be used
	bool useDepth;
	// specifies if the params have been initialized
	bool paramInit;
	// specifies if the detector has been initialized
	bool detectorInit;
	// specifies the number of bins to be used for histogram - for each channel
	int noOfBins[4];
	// defines range for histograms
	float histRange[4][2];//, histRange1[2], histRange2[2];
	// specifies the size of the input frame
	Size frameSize;

	// container for histograms RGB-D
	MatND hist[4];
	// containers for backprojected images
	Mat backPro[4];

	/*-----------------Member functions-----------------------*/
	// function to create color model - histogram models
	void createColorModel(Mat & _rgbImg, Mat & _depthImg, Mat & _mask);

public:
	// default destructor
	virtual ~HDcolorModel() { }
	// default constructor
	HDcolorModel(void);
	// constructor with noOfBins specified
	HDcolorModel(vector<int> noOfBins, bool _useDepth);

	// constructor to initialize the detector object
	virtual bool initialize(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool _useDepth);
	// actual function to detect hand - right now just gives probability image - might be changed to bounding box output
	virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg);
	// function to get param values
	virtual void getParams(vector<int> intParams, vector<double> doubleParams) const;
	// function to set param values
	virtual void setParams(vector<int> intParams, vector<double> doubleParams);
};

}
}
#endif
