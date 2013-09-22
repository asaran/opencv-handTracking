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

// Author - Gurpinder Singh Sandhu

#ifndef _HDHISTBACKPROJ_HPP_
#define _HDHISTBACKPROJ_HPP_

#include <opencv2/contrib/hand_detector.hpp>
#include <vector>

using namespace std;

namespace cv {
namespace HT {

class CV_EXPORTS HistBackProj: public HandDetector {
public:
    // Structure for storing parameters required
    struct CV_EXPORTS_W_SIMPLE Params {
        // specifies the number of bins to be used for histogram - for each channel
        CV_PROP_RW int noOfBins[4];
        // defines range for histograms
        CV_PROP_RW float histRange[4][2];
        // specifies the size of the input frame
        //CV_PROP_RW Size frameSize;
        // color code used for conversion ; use -1 for no conversion ; default COLOR_BGR2HSV
        CV_PROP_RW int colorCode;
        // specifies if the depth is to be used
        CV_PROP_RW bool useDepth;
        // specifies if the color is to be used
        CV_PROP_RW bool useColor;

        CV_WRAP Params();

        void read( const FileStorage& fn );
        void write( FileStorage& fs ) const;
    };

    /*-----------------Member functions-----------------------*/
    // default destructor
    virtual ~HistBackProj() { }
    // default constructor
    CV_WRAP HistBackProj(const Params &parameters = Params());

    // constructor to initialize the detector object
    virtual bool train(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental = false);
    // actual function to detect hand - right now just gives probability image - might be changed to bounding box output
    virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg);
    // load model from xml file
    virtual bool load(vector<String> &fileNamePrefix);
    // save model to a file
    virtual bool save(vector<String> &fileNamePrefix);

protected:
    // specifies if the params have been initialized
    //bool paramInit;
    // specifies if the detector has been initialized
    bool detectorInit;
    // Parameters for list
    Params params;
    // container for histograms RGB-D
    MatND hist[4];
    MatND histTemp[4];
    // no. of Images trained on till now
    unsigned int noOfImages;
    // containers for backprojected images
    Mat backPro[4];
    // image for internal calc
    Mat img;
    // Output Img
    Mat probImg;


    /*-----------------Member functions-----------------------*/
    // function to create color model - histogram models
    void createColorModel(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental);
};

}
}
#endif

