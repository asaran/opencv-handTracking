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
//              - Cheng Li (All Lc.. entities)

#include <precomp.hpp>
#include "opencv2/contrib/hd_per_pix.hpp"
#include <sstream>

namespace cv {
namespace HT {

PerPixRegression::Params::Params() {
    knn = 1;
    numModels = 1;
    models = 0;
    featureString = "rvl";
}

void PerPixRegression::initialiseFLANN(void) {
    searchTree  = *new flann::Index(histAll, indexParams);
    flannInit = true;
}

void PerPixRegression::test(Mat &img, int num_models, OutputArray probImg)
{
    if(num_models > param.knn) return;
    Mat hist;
    computeColorHist_HSV(img,hist);                                 // extract hist
    indices.clear();
    searchTree.knnSearch(hist, indices, dists, param.knn);            // probe search
    Mat descriptors;
    extractor.work(img,descriptors,3,&kp);

    if(!responseAvg.data) responseAvg = Mat::zeros(descriptors.rows,1,CV_32FC1);
    else responseAvg *= 0;

    int n = descriptors.rows;


    float norm = 0;
    for(int i=0;i<num_models;i++)
    {
        int idx = indices[i];
        responseVec = Mat::zeros(n, 1, CV_32FC1);
        for(int k=0; k<n; k++)
            responseVec.at<float>(k,0) = classifier[idx]->predict(descriptors.row(k));       // run classifier

        responseAvg += responseVec*float(pow(0.9f,(float)i));
        norm += float(pow(0.9f,(float)i));
    }

    responseAvg /= norm;

    sz = img.size();
    bs = extractor.bound_setting;
    rasterizeResVec(responseImg,responseAvg,kp,sz);       // class one

    vector<Point2f> pt;
    postProcessImg = postprocess(responseImg,pt);
    //colormap(postProcessImg,postProcessImg,1);
    //cvtColor(_ppr,_ppr,CV_GRAY2BGR);

    postProcessImg.copyTo(probImg);
}

Mat PerPixRegression::postprocess(Mat &img,vector<Point2f> &pt)
{
    Mat tmp;
    GaussianBlur(img,tmp,Size(31,31),0,0,BORDER_REFLECT);
    
    tmp = tmp > 0.04;

    vector<vector<Point> > co;
    vector<Vec4i> hi;

    findContours(tmp,co,hi,RETR_EXTERNAL,CHAIN_APPROX_NONE);
    tmp *= 0;

    Moments m;
    for(int i=0;i<(int)co.size();i++)
    {
        if(contourArea(Mat(co[i])) < 300) continue;
        drawContours(tmp, co,i, CV_RGB(255,255,255), CV_FILLED, CV_AA);
        m = moments(Mat(co[i]));
        pt.push_back(Point2f(m.m10/m.m00,m.m01/m.m00));
    }

    return tmp;

}

void PerPixRegression::rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, Size s)
{
    if((img.rows!=s.height) || (img.cols!=s.width) || (img.type()!=CV_32FC1) ) img = Mat::zeros( s, CV_32FC1);

    for(int i = 0;i< (int)keypts.size();i++)
    {
        int r = floor(keypts[i].pt.y);
        int c = floor(keypts[i].pt.x);
        img.at<float>(r,c) = res.at<float>(i,0);
    }
}

void PerPixRegression::computeColorHist_HSV(Mat &src, Mat &hist)
{

    int bins[] = {4,4,4};
    if(src.channels()!=3) exit(1);

    Mat hsv;
    cvtColor(src,hsv,COLOR_BGR2HSV_FULL);

    int histSize[] = {bins[0], bins[1], bins[2]};
    Mat his;
    his.create(3, histSize, CV_32F);
    his = Scalar(0);
    CV_Assert(hsv.type() == CV_8UC3);
    MatConstIterator_<Vec3b> it = hsv.begin<Vec3b>();
    MatConstIterator_<Vec3b> it_end = hsv.end<Vec3b>();
    for( ; it != it_end; ++it )
    {
        const Vec3b& pix = *it;
        his.at<float>(pix[0]*bins[0]/256, pix[1]*bins[1]/256,pix[2]*bins[2]/256) += 1.f;
    }

    // ==== Remove small values ==== //
    float minProb = 0.01;
    minProb *= hsv.rows*hsv.cols;
    Mat plane;
    const Mat *_his = &his;

    NAryMatIterator itt = NAryMatIterator(&_his, &plane, 1);
    threshold(itt.planes[0], itt.planes[0], minProb, 0, THRESH_TOZERO);
    double s = sum(itt.planes[0])[0];

    // ==== Normalize (L1) ==== //
    s = 1./s * 255.;
    itt.planes[0] *= s;
    itt.planes[0].copyTo(hist);


}

PerPixRegression::PerPixRegression(const Params &parameters) : param(parameters) {
    RTparams.max_depth               = 10;
    RTparams.regression_accuracy     = 0.00f;
    RTparams.min_sample_count        = 10;

    bs = 0;
    flannInit = false;
    featureInit = false;
    classifierInit = false;

    indexParams = *new flann::KMeansIndexParams;
    indices    = vector<int> (param.knn);
    dists      = vector<float> (param.knn);

    classifier.clear();
}

bool PerPixRegression::train(Mat &_rgbImg, Mat &_depthImg, Mat &_mask, bool incremental) {
    //Extract histogram
    if(!featureInit) {
        extractor.set_extractor(param.featureString);
        featureInit = true;
    }
    Mat hist;
    computeColorHist_HSV(_rgbImg,hist);
    if(!incremental)
        histAll.release();

    //push to histAll
    histAll.push_back(hist);

    Mat desc;
    Mat lab;
    vector<KeyPoint> kp;
    CvRTrees *rt = new CvRTrees;

    _mask.convertTo(_mask,CV_8UC1);
    extractor.work(_rgbImg, desc, _mask, lab, 3, &kp);

    Mat varType = Mat::ones(desc.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL;
    rt->train(desc,CV_ROW_SAMPLE,lab,Mat(),Mat(),varType,Mat(), RTparams);

    if(!incremental) {
        param.models = 0;
        classifier.clear();
    }
    classifier.push_back(rt);
    param.models++;
    classifierInit = true;
    return true;
}

void PerPixRegression::detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg) {
    CV_Assert(classifierInit == true && "Classifiers not trained/loaded yet");
    if(!featureInit) {
            extractor.set_extractor(param.featureString);
            featureInit = true;
    }
    if(!flannInit)
        initialiseFLANN();
    //Mat temp = Mat();
    test(_rgbImg, param.numModels, probImg);
}

bool PerPixRegression::save(vector<String> &fileNamePrefix) {
    CV_Assert(param.models > 0 && "No models to save");
    CV_Assert(fileNamePrefix.size() == 3);
    FileStorage fs;
    CV_Assert(fs.open(fileNamePrefix[0] + ".xml", FileStorage::WRITE) && "Could not open the file storage. Check the path and permissions");
    fs << "numOfModels" << param.models;
    fs << "featureString" << param.featureString;
    fs.release();

    stringstream s1, s2;

    for(int i=0; i<param.models; i++) {
        s1.str("");
        s1 << fileNamePrefix[1].c_str() << i << ".xml";
        s2.str("");
        s2 << fileNamePrefix[2].c_str() << i << ".xml";
        CV_Assert(fs.open(s1.str().c_str(), FileStorage::WRITE) && "Could not open the file storage. Check the path and permissions");
        fs << "hsv" << histAll.row(i);
        fs.release();
        classifier[i]->save(s2.str().c_str());
    }
    return true;
}

bool PerPixRegression::load(vector<String> &fileNamePrefix) {
    histAll.release();
    classifier.clear();
    CV_Assert(fileNamePrefix.size() == 3);
    FileStorage fs;
    CV_Assert(fs.open(fileNamePrefix[0] + ".xml", FileStorage::READ) && "Could not open the file storage. Check the path and permissions");
    fs["numOfModels"] >> param.models;
    fs["featureString"] >> param.featureString;
    fs.release();
    classifier = vector<CvRTrees*>(param.models);

    Mat hist;
    stringstream s1, s2;

    for(int i=0; i<param.models; i++) {
        s1.str("");
        s1 << fileNamePrefix[1].c_str() << i << ".xml";
        s2.str("");
        s2 << fileNamePrefix[2].c_str() << i << ".xml";
        CV_Assert(fs.open(s1.str().c_str(), FileStorage::READ) && "Could not open the file storage. Check the path and permissions");

        fs["hsv"] >> hist;
        histAll.push_back(hist);
        fs.release();
        classifier[i] = new CvRTrees;
        classifier[i]->load(s2.str().c_str());
    }
    classifierInit = true;
    return true;
}

// Following has been written by Cheng Li

enum FeatureExtractorType{
    FEAT_RGB,
    FEAT_HSV,
    FEAT_LAB,
    FEAT_HOG,
    FEAT_SIFT,
    FEAT_SURF,
    FEAT_BRIEF,
    FEAT_ORB
};


void LcFeatureExtractor::set_extractor( String setting_string )
{
    int bo[100]; memset(bo,0,sizeof(int)*100);

    computers.clear();

    for(int i = 0;i<(int) setting_string.length();i++)
    {
        switch(setting_string[i])
        {
            case 's':
                bo[FEAT_SIFT] = 1;
                break;
            case 'h':
                bo[FEAT_HOG] = 1;
                break;
            case 'l':
                bo[FEAT_LAB]   = 1;
                break;
            case 'v':
                bo[FEAT_HSV]   = 1;
                break;
            case 'b':
                bo[FEAT_BRIEF] = 1;
                break;
            case 'o':
                bo[FEAT_ORB]   = 1;
                break;
            case 'r':
                bo[FEAT_RGB] = 1;
                break;
            case 'u':
                bo[FEAT_SURF] = 1;
                break;
        }
    }

    if(bo[FEAT_RGB])
    {
        computers.push_back( new LcColorComputer< LC_RGB,1> );
        computers.push_back( new LcColorComputer< LC_RGB,3> );
        computers.push_back( new LcColorComputer< LC_RGB,5> );
    }

    if(bo[FEAT_HSV])
    {
        computers.push_back( new LcColorComputer< LC_HSV,1> );
        computers.push_back( new LcColorComputer< LC_HSV,3> );
        computers.push_back( new LcColorComputer< LC_HSV,5> );
    }

    if(bo[FEAT_LAB])
    {
        computers.push_back( new LcColorComputer< LC_LAB,1> );
        computers.push_back( new LcColorComputer< LC_LAB,3> );
        computers.push_back( new LcColorComputer< LC_LAB,5> );
    }

    if(bo[FEAT_HOG])
    {
        computers.push_back( new LcHoGComputer );
    }

    if(bo[FEAT_SIFT])
    {
        computers.push_back( new LcSIFTComputer );
    }

    if(bo[FEAT_SURF])
    {
        computers.push_back( new LcSURFComputer );
    }

    if(bo[FEAT_BRIEF])
    {
        computers.push_back( new LcBRIEFComputer );
    }

    if(bo[FEAT_ORB])
    {
        computers.push_back( new LcOrbComputer );
    }

    for(int i = 0;i<(int)computers.size();i++) computers[i]->veb = veb;
}


LcFeatureExtractor::LcFeatureExtractor()
{
    veb = 0;

    bound_setting = 0;

    computers.clear();

    //set_extractor( "robvltdmchsug" );

    set_extractor( "rl" );

    for(int i = 0;i< (int)computers.size();i++) computers[i]->veb = veb;

}

void LcFeatureExtractor::work(Mat & img, Mat & desc, vector<KeyPoint> * p_keypoint)
{
    Mat temp = Mat();
    work( img, desc, temp , temp ,1, p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc,int step_size, vector<KeyPoint> * p_keypoint)
{
    Mat temp = Mat();
    work( img, desc, temp , temp ,step_size, p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab,vector<KeyPoint> * p_keypoint)
{
    int step_size = 1;
    work(img,desc,img_gt,lab,step_size,p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, int step_size, vector<KeyPoint> * p_keypoint)
{

    //cout << "work" << endl;

    vector<KeyPoint> * mp_keypts;

    if( p_keypoint == NULL)     mp_keypts = new vector<KeyPoint>;
    else                        mp_keypts = p_keypoint;

    vector<KeyPoint> & keypts = *mp_keypts;
    vector<KeyPoint> keypts_ext;

    Mat img_ext;

    img2keypts(img, keypts, img_ext, keypts_ext, step_size);                // set keypoints

    int dims = get_dim();

    allocate_memory(desc,dims, (int) keypts.size());

    extract_feature( img, keypts, img_ext, keypts_ext, desc);

    if(img_gt.data) Groundtruth2Label( img_gt, img.size(), keypts, lab);

}

void LcFeatureExtractor::Groundtruth2Label( Mat & img_gt, Size _size , vector< KeyPoint>  keypts , Mat & lab)
{

    Mat im;

    resize(img_gt ,im,_size ,0,0,INTER_NEAREST);

    lab = Mat::zeros((int)keypts.size(),1,CV_32FC1);
    for(int i=0;i<(int)keypts.size();i++)
    {
        Point p ((int)floor(.5+keypts[i].pt.x),(int)floor(.5+keypts[i].pt.y) );

        if((int)im.at<uchar>(p.y,p.x)>100) lab.at<float>(i,0) = 0.5;            // don't care
        if((int)im.at<uchar>(p.y,p.x)>200) lab.at<float>(i,0) = 1.0;            // ground truth
    }

    if(veb) cout << " label size " << lab.rows << " by " << lab.cols << endl;
}

void LcFeatureExtractor::img2keypts(
                        Mat & img, vector<KeyPoint> & keypts,
                        Mat & img_ext, vector<KeyPoint> & keypts_ext,
                        int step_size)
{

    int bound_max = get_maximal_bound();

    if(bound_setting<0) bound_setting = bound_max;

    DenseFeatureDetector dfd;

    float   initFeatureScale    = 1.f;              // inital size
    int     featureScaleLevels  = 1;                // one level
    float   featureScaleMul     = 1.00f;            // multiplier (ignored if only one level)
    int     train_initXyStep    = step_size;        // space between pixels for training (must be 1)

    dfd = DenseFeatureDetector(initFeatureScale,featureScaleLevels,featureScaleMul,train_initXyStep,bound_setting,true,false);

    dfd.detect(img,keypts);

    // bound_setting must be adjusted depending on even and odd ?

    if( bound_max > bound_setting)
    {

        DenseFeatureDetector dfd_ext;

        int diff = bound_max-bound_setting;

        img_ext = Mat::zeros( img.rows + diff*2 , img.cols + diff*2, img.type());

        img.copyTo( img_ext( Range(diff, diff+ img.rows),Range(diff, diff+ img.cols) ) );


        dfd = DenseFeatureDetector(initFeatureScale,featureScaleLevels,featureScaleMul,train_initXyStep,bound_max,true,false);

        dfd.detect(img_ext,keypts_ext);
    }
}

int LcFeatureExtractor::get_maximal_bound()
{
    int ans = 0;
    for(int i = 0;i< (int)computers.size();i++) ans = max(ans, computers[i]->bound);
    return ans;
}

void LcFeatureExtractor::allocate_memory(Mat & desc,int dims,int data_n)
{
    desc = Mat::zeros(data_n,dims ,CV_32FC1);
}

void LcFeatureExtractor::extract_feature(
    Mat & img,vector<KeyPoint> & keypts,
    Mat & img_ext, vector<KeyPoint> & keypts_ext,
    Mat & desc)
{
    int data_n = (int)keypts.size();

    int d = 0;
    for(int i = 0;i< (int)computers.size();i++)
    {
        int dim = computers[i]->dim;
        Mat _desc = desc(Rect(d,0, dim ,data_n));

        if( computers[i]->bound < bound_setting ) computers[i]->compute( img, keypts, _desc);
        else computers[i]->compute( img_ext , keypts_ext , _desc);
        d+= dim;
    }
}

int LcFeatureExtractor::get_dim()
{
    int ans = 0;

    for(int i = 0;i< (int)computers.size();i++)
    {
        ans+= computers[i]->dim;
    }
    return ans;
}

//=============================

template< ColorSpaceType color_type, int win_size>
LcColorComputer<color_type, win_size>::LcColorComputer()
{
    if(win_size ==1) dim = 3;
    else
    {
        dim = win_size*win_size - (win_size-2)*(win_size-2);
        dim = dim*3;
    }
    bound = (win_size-1)/2;
}

template< ColorSpaceType color_type, int win_size>
void LcColorComputer<color_type, win_size>::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
    int code;
    if(color_type==LC_RGB) code = COLOR_BGR2RGB;
    else if(color_type==LC_HSV) code = COLOR_BGR2HSV_FULL;
    else if(color_type==LC_LAB) code = COLOR_BGR2Lab;

    Mat color;
    if(color_type != LC_RGB)
        cvtColor(src,color,code);
    else
        src.copyTo(color);

    for(int k=0;k<(int)keypts.size();k++)
    {
        int r = int(floor(.5+keypts[k].pt.y) - floor(win_size*0.5));    // upper-left of patch
        int c = int(floor(.5+keypts[k].pt.x) - floor(win_size*0.5));
        int a = 0;

        for(int i=0;i<win_size;i++)
        {
            for(int j=0;j<win_size;j++)
            {
                if(i==0 || j==0 || i==win_size-1 || j == win_size-1)
                {
                    desc.at<float>(k,a+0) = color.at<Vec3b>(r+i,c+j)(0)/255.f;
                    desc.at<float>(k,a+1) = color.at<Vec3b>(r+i,c+j)(1)/255.f;
                    desc.at<float>(k,a+2) = color.at<Vec3b>(r+i,c+j)(2)/255.f;
                    a+=3;
                }
            }
        }
    }
}


//==========================


//==========================

LcHoGComputer::LcHoGComputer()
{
    dim = 36;
    bound = 10;
}


void LcHoGComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
    HOGDescriptor hog(src.size(),Size(16,16),Size(1,1),Size(8,8),9);

    vector<float> hog_desp;


    hog_desp.clear();

    Mat try_src =  Mat::zeros( src.size(),CV_8U);

    hog.compute(src,hog_desp);

    int block_size = 16;

    int shift_size = (block_size+1)/2;

    int rows_step = src.rows-block_size+1;

    int HOG_dim = 36;


    for(int k=0;k<(int)keypts.size();k++)
    {
        int r = (int)floor(.5+keypts[k].pt.y);
        int c = (int)floor(.5+keypts[k].pt.x);

        int id_start = (rows_step * (c-shift_size) + r-shift_size)*HOG_dim;
        for(int i = 0;i<HOG_dim;i++)
        {
            desc.at<float>(k,i) = hog_desp[id_start];
            id_start++;

        }

    }
}

//===============================

LcBRIEFComputer::LcBRIEFComputer()
{
    dim = 16;
    bound = 28;
}


void LcBRIEFComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
    Mat brief_desc;
    BriefDescriptorExtractor bde(16);
    bde.compute(src,keypts,brief_desc);
    brief_desc.convertTo(brief_desc,CV_32FC1);

    for(int i=0;i<(int)brief_desc.rows;i++){
        Mat row = brief_desc.row(i);
        normalize(row,row,1.0,0,NORM_L1);
    }

    brief_desc.copyTo(desc);
}

//===============================

LcSIFTComputer::LcSIFTComputer()
{
    dim = 128;
    bound = 0;
}


void LcSIFTComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
    //String gp = "SIFT";
    //Ptr<DescriptorExtractor> sift1 = DescriptorExtractor::create(gp);
    SIFT sift;
    Mat sift_desc;
    vector<KeyPoint> kk = keypts;
    sift( src, cv::noArray() , keypts,sift_desc,true);
    sift_desc.convertTo(sift_desc,5);

    for(int i=0;i<(int)sift_desc.rows;i++){
        Mat row = sift_desc.row(i);
        normalize(row,row,1.0,0,NORM_L1);
    }
    sift_desc.copyTo(desc);
}

//===============================

LcSURFComputer::LcSURFComputer()
{
    dim = 128;
    bound = 0;
}


void LcSURFComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{

    //Ptr<DescriptorExtractor> surf = DescriptorExtractor::create("SURF");
    SURF surf;
    Mat surf_desc;
    surf( src, Mat(), keypts,surf_desc,true);
    surf_desc.convertTo(surf_desc,CV_32FC1);

    for(int i=0;i<(int)surf_desc.rows;i++){
        Mat row = surf_desc.row(i);
        normalize(row,row,1.0,0,NORM_L1);
    }
    surf_desc.copyTo(desc);
}

//===============================


//===============


LcOrbComputer::LcOrbComputer()
{
    dim = 32;
    bound = 31;
}


void LcOrbComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
    Mat orb_desc;
    OrbDescriptorExtractor ode;
    ode.compute(src,keypts,orb_desc);
    orb_desc.convertTo(orb_desc,CV_32FC1);

    for(int i=0;i<(int)orb_desc.rows;i++){
        Mat row = orb_desc.row(i);
        normalize(row,row,1.0,0,NORM_L1);
    }

    orb_desc.copyTo(desc);
}
}
}

