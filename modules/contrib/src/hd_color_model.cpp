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

const int def_noOfBins[] = {32, 32, 32};
const bool def_useDepth = false;
float def_hranges[] = {0, 180};
float def_sranges[] = {0, 256};
float def_vranges[] = {0, 256};
float def_dranges[] = {400, 7000};     // approx. to be replaced with exact value

HistBackProj::HistBackProj(const HistBackProj::Params &parameters) : params(parameters) {
    //useColor = true;
    //useDepth = false;
    //paramInit = true;
    CV_Assert(params.useColor || params.useDepth);
    noOfImages = 0;
    detectorInit = false;
}


// constructor for Params
HistBackProj::Params::Params() {
    for(int i=0; i<3; i++) {
        noOfBins[i] = def_noOfBins[i];
    }
    noOfBins[3] = 0;
    histRange[0][0] = def_hranges[0];
    histRange[0][1] = def_hranges[1];
    histRange[1][0] = def_sranges[0];
    histRange[1][1] = def_sranges[1];
    histRange[2][0] = def_vranges[0];
    histRange[2][1] = def_vranges[1];
    histRange[3][0] = def_dranges[0];
    histRange[3][1] = def_dranges[1];
    colorCode = COLOR_BGR2HSV;
    useColor = true;
    useDepth = false;
}

// read params from a file
void HistBackProj::Params::read(const FileStorage& fn) {
    noOfBins[0] = fn["noOfBins[0]"];
    noOfBins[1] = fn["noOfBins[1]"];
    noOfBins[2] = fn["noOfBins[2]"];
    noOfBins[3] = fn["noOfBins[3]"];

    histRange[0][0] = fn["histRange[0][0]"];
    histRange[0][1] = fn["histRange[0][1]"];
    histRange[1][0] = fn["histRange[1][0]"];
    histRange[1][1] = fn["histRange[1][1]"];
    histRange[2][0] = fn["histRange[2][0]"];
    histRange[2][1] = fn["histRange[2][1]"];
    histRange[3][0] = fn["histRange[3][0]"];
    histRange[3][1] = fn["histRange[3][1]"];

    colorCode = fn["colorCode"];
    useColor = (int)fn["useColor"] > 0 ? true : false;
    useDepth = (int)fn["useDepth"] > 0 ? true : false;
}

// write params to a file
void HistBackProj::Params::write( FileStorage& fs ) const {
    fs << "noOfBins[0]" << noOfBins[0];
    fs << "noOfBins[1]" << noOfBins[1];
    fs << "noOfBins[2]" << noOfBins[2];
    fs << "noOfBins[3]" << noOfBins[3];

    fs << "histRange[0][0]" << histRange[0][0];
    fs << "histRange[0][1]" << histRange[0][1];
    fs << "histRange[1][0]" << histRange[1][0];
    fs << "histRange[1][1]" << histRange[1][1];
    fs << "histRange[2][0]" << histRange[2][0];
    fs << "histRange[2][1]" << histRange[2][1];
    fs << "histRange[3][0]" << histRange[3][0];
    fs << "histRange[3][1]" << histRange[3][1];

    fs << "colorCode" << colorCode;
    fs << "useColor" << (int)useColor;
    fs << "useDepth" << (int)useDepth;
}

// training method for HistBackProj. Calculates histograms using the input data and averages out the histograms
// if incremental is true
bool HistBackProj::train(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental) {
    if(params.useColor) {
        CV_Assert(_rgbImg.type() == CV_8UC3 && _mask.type() == CV_8UC1);
    }
    if(params.useDepth) {
        CV_Assert(_depthImg.type() == CV_16UC1);
        CV_Assert(_rgbImg.size() == _depthImg.size());
    }

    //if(paramInit) {
    createColorModel(_rgbImg, _depthImg, _mask, incremental);
    //}
    probImg = Mat(_rgbImg.size(), CV_32F, Scalar(0.0));
    return detectorInit;
}

void HistBackProj::detect(Mat & _rgbImg, Mat & _depthImg, OutputArray _probImg) {
    CV_Assert(detectorInit == true && "Detector not initialized yet. Please call train(...) first\n");

    if(params.colorCode < 0)
        _rgbImg.copyTo(img);
    else
        cvtColor(_rgbImg, img, params.colorCode);

    const float* range[] = {params.histRange[0], params.histRange[1], params.histRange[2], params.histRange[3]};

    if(params.useColor) {
        vector<Mat> channel;
        split(img, channel);
        calcBackProject(&channel[0], 1, 0, hist[0], backPro[0], range);
        calcBackProject(&channel[1], 1, 0, hist[1], backPro[1], range+1);
        calcBackProject(&channel[2], 1, 0, hist[2], backPro[2], range+2);

        multiply(backPro[0], backPro[1], probImg, 1./255.0, CV_32F);
        multiply(backPro[2], probImg, probImg, 1./255.0, CV_32F);
    }

    if(params.useDepth) {
        calcBackProject(&_depthImg, 1, 0, hist[3], backPro[3], range+3);
        if(params.useColor)
            multiply(backPro[3], probImg, probImg, 1./255.0, CV_32F);
        else
            backPro[3].convertTo(probImg, CV_32F);
    }
    probImg.convertTo(_probImg, CV_8U);
}

bool HistBackProj::load(const String &fileNamePrefix) {
    FileStorage fs;
    CV_Assert(fs.open(fileNamePrefix + "info.xml", FileStorage::READ) && "Could not open file");
    params.read(fs);
    fs.release();

    CV_Assert(fs.open(fileNamePrefix + "hist.xml", FileStorage::READ) && "Could not open file");
    if(params.useColor) {
        fs["hist[0]"] >> hist[0];
        fs["hist[1]"] >> hist[1];
        fs["hist[2]"] >> hist[2];
    }
    if(params.useDepth)
        fs["hist[3]"] >> hist[3];
    return true;
}

bool HistBackProj::save(const String &fileNamePrefix) {
    FileStorage fs;
    CV_Assert(fs.open(fileNamePrefix + "info.xml", FileStorage::WRITE) && "Could not open file");
    params.write(fs);
    fs.release();

    CV_Assert(fs.open(fileNamePrefix + "hist.xml", FileStorage::WRITE) && "Could not open file");
    if(params.useColor) {
        fs << "hist[0]" << hist[0];
        fs << "hist[1]" << hist[1];
        fs << "hist[2]" << hist[2];
    }
    if(params.useDepth)
        fs << "hist[3]" << hist[3];
    return true;
}

void HistBackProj::createColorModel(Mat &_rgbImg, Mat & _depthImg, Mat & _mask, bool incremental) {
    if(params.colorCode < 0)
        _rgbImg.copyTo(img);
    else
        cvtColor(_rgbImg, img, params.colorCode);

    const float* range[] = {params.histRange[0], params.histRange[1], params.histRange[2], params.histRange[3]};

    if(params.useColor) {
        vector<Mat> channel;
        split(img, channel);
        calcHist(&channel[0], 1, 0, _mask, histTemp[0], 1, &params.noOfBins[0], range);
        calcHist(&channel[1], 1, 0, _mask, histTemp[1], 1, &params.noOfBins[1], range+1);
        calcHist(&channel[2], 1, 0, _mask, histTemp[2], 1, &params.noOfBins[2], range+2);
    }

    if(params.useDepth) {
        calcHist(&_depthImg, 1, 0, _mask, histTemp[3], 1, &params.noOfBins[2], range+3);
    }

    if(!incremental || noOfImages == 0) {
        for(int i=0; i<4; i++)
            histTemp[i].copyTo(hist[i]);
        noOfImages = 1;
    }
    else {
        for(int i=0; i<4; i++) {
            hist[i] = ((float)noOfImages*hist[i] + histTemp[i]);
            hist[i] = hist[i]/(float)(noOfImages+1);
        }
        noOfImages++;
    }
    detectorInit = true;
}
}
}


