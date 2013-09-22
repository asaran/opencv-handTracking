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

#ifndef _HANDDETECTOR_HPP_
#define _HANDDETECTOR_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/nonfree.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

namespace cv {
namespace HT {

//Base class for Detector module. Defines bare minimum for derived classes to provide common interface

//Initialization of all detectors must be done using a mask which the user has to provide. */

class CV_EXPORTS HandDetector {
protected:

public:
    //function to create hand detector of any type
    //CV_WRAP static Ptr<HandDetector> create( const String& detectorType );
    //train method - returns true if training successful. Each detector class may define overloaded functions according to different needs.
    virtual bool train(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental) = 0;
    //Detect function to be called in the video loop for subsequent detection of the hand
    virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg) = 0;
    //Load a detector from a file
    virtual bool load(vector<String> &fileNamePrefix) = 0;
    //Save a detector to a file
    virtual bool save(vector<String> &fileNamePrefix) = 0;
    //Virtual Destructor for HandDetector class
    virtual ~HandDetector() { }
};

}
}
#endif

