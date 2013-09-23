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

// Contributers - Gurpinder Singh Sandhu
//              - Cheng Li
//              - Kris Kitani


#ifndef _HDPERPIXREGRESSION_HPP_
#define _HDPERPIXREGRESSION_HPP_

#include <opencv2/contrib/hand_detector.hpp>
#include <fstream>

using namespace std;

namespace cv {
namespace HT {

class LcFeatureComputer
{
public:
    int dim;
    int bound;
    int veb;
    bool use_motion;
    virtual void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc) = 0;
    virtual ~LcFeatureComputer(){}
};



class LcFeatureExtractor
{
public:

    LcFeatureExtractor();

    int veb;

    int bound_setting;

    void img2keypts( Mat & img, vector<KeyPoint> & keypts,Mat & img_ext, vector<KeyPoint> & keypts_ext, int step_size);

    void work(Mat & img , Mat & desc, int step_size, vector<KeyPoint> * p_keypoint = NULL);

    void work(Mat & img , Mat & desc, vector<KeyPoint> * p_keypoint = NULL);
    // That's the main interface member which return a descriptor matrix
    // with an image input

    void work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, vector<KeyPoint> * p_keypoint = NULL);
    //with ground truth image output at same time

    void work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, int step_size, vector<KeyPoint> * p_keypoint = NULL);


    void set_extractor( String setting_string );

private:

    vector < LcFeatureComputer * > computers;

    int get_dim();

    int get_maximal_bound();

    void allocate_memory(Mat & desc,int dims,int data_n);

    void extract_feature( Mat & img,vector<KeyPoint> & keypts,
                        Mat & img_ext, vector<KeyPoint> & keypts_ext,
                        Mat & desc);

    void Groundtruth2Label( Mat & img_gt, Size _size , vector< KeyPoint> , Mat & lab);

};

//================

enum ColorSpaceType{
    LC_RGB,LC_LAB,LC_HSV
};

template< ColorSpaceType color_type, int win_size>
class LcColorComputer: public LcFeatureComputer
{
public:
    LcColorComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};


//================

class LcHoGComputer: public LcFeatureComputer
{
public:
    LcHoGComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//================

class LcBRIEFComputer: public LcFeatureComputer
{
public:
    LcBRIEFComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//===================

class LcSIFTComputer: public LcFeatureComputer
{
public:
    LcSIFTComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//===================

class LcSURFComputer: public LcFeatureComputer
{
public:
    LcSURFComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};


//====================

class LcOrbComputer: public LcFeatureComputer
{
public:
    LcOrbComputer();
    void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};


/////////////perPixRegression////////////////////

class CV_EXPORTS PerPixRegression : public HandDetector {
public :
    struct CV_EXPORTS_W_SIMPLE Params {
        // specifies number of nearest neighbours for initialising the classifier
        CV_PROP_RW int knn;
        // specifies the number of number of nearest neighbours for classification
        CV_PROP_RW int numModels;
        // specifies the number of classifiers trained so far
        CV_PROP_RW int models;
        // specifies which features to use
        CV_PROP_RW String featureString;
        // specifies step size for considering pixels for processing - training
        CV_PROP_RW int training_step_size;
        // specifies step size for considering pixels for proccesing - testing/detecting
        CV_PROP_RW int testing_step_size;

        Params();
        void read( const FileStorage& fn );
        void write( FileStorage& fs ) const;
    };

    // constructor for PerPixRegression
        CV_WRAP PerPixRegression(const Params &parameters = Params());
        // Default destructor
        virtual ~PerPixRegression() { 
            for(int i=0; i<param.models; i++)
                delete classifier[i];
        }

        // Function to train models on input image
        virtual bool train(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental = false);
        // Function to detect images - outputs binary image
        virtual void detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg);
        // save trained models with general configuration file with configFileName, global feature files with featureFilePrefix, models with modelFilePrefix in that order in a vector. All names without .xml
        virtual bool save(vector<String> &fileNamePrefix);
        // load classifier from saved files - symmetric to load
        virtual bool load(vector<String> &fileNamePrefix);

protected :
    int bs;
    Size sz;
    Params param;
    vector<int> indices;
    vector<float> dists;
    Mat                         descriptors;
    vector<KeyPoint>            kp;
    Mat                         responseAvg;
    Mat                         responseVec;
    Mat                         responseImg;
    Mat                         rawImg;
    Mat                         bluImg;
    Mat                         postProcessImg;               // post processed


    ///------------------///
    //classifier
    vector<CvRTrees*>           classifier;
    // search tree
    flann::Index                searchTree;
    // flann index params
    flann::IndexParams          indexParams;
    // feature extractor
    LcFeatureExtractor          extractor;
    // object storing hsv features for matching
    Mat                         histAll;              // do not destroy!
    // random tree params
    CvRTParams RTparams;

    // variables specifying intializing of modules
    bool flannInit;
    bool featureInit;
    bool classifierInit;
    
    /*-------------------Member Functions---------------*/
    // Function for testing image (i.e detection)
    void test(Mat &img,int num_models, OutputArray probImg);
    // Function to convert output from vector of points to image
    Mat postprocess(Mat &img,vector<Point2f> &pt);

    // Function to compute color Histogram
    void computeColorHist_HSV(Mat &src, Mat &hist);
    // Funtion to raterise result vector
    void rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, Size s);
    // initialize nearest neighbour search
    void initialiseFLANN(void);
};
}
}
#endif

