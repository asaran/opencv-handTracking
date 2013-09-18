#include <opencv2/contrib/hd_per_pix.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// function to convert a text file of image paths to a vector of paths
void imagePathLoader(const char *fileName, vector<String> &list, int n);
// function to perform general preprocessing steps
void preprocess(Mat &img, Mat &maskImg);


int main(){
    HT::PerPixRegression dt;
    String configFilePrefix = "/home/guru/trained2/config";
    String featureFilePrefix = "/home/guru/trained2/features/feature";
    String modelFilePrefix = "/home/guru/trained2/models/model";


    // list of image and mask file paths (with file type)
    vector<String> imgPaths;
    vector<String> maskPaths;

    string imgNameFilePath = "/home/guru/trained2/imgFile.txt";
    string maskNameFilePath = "/home/guru/trained2/maskFile.txt";
    // no of images to train for
    int noOfImages = 10;
    // load image paths
    imagePathLoader(imgNameFilePath.c_str(), imgPaths, noOfImages);
    // load mask paths
    imagePathLoader(maskNameFilePath.c_str(), maskPaths, noOfImages);



////////////////// Main sample codes starts ///////////////////////


    // No. of Images and Masks should be equal
    CV_Assert(imgPaths.size() == maskPaths.size());

    // Mat object for bgr and mask images
    Mat img, maskImg;

    // For perpix depthImg is not required as such (right now the code doesnot uses depth)
    Mat depthImg = Mat();

    int N_img = imgPaths.size();

    for(int i=0; i<N_img; i++) {
        //read images and masks
        img = imread(imgPaths[i]);
        maskImg = imread(maskPaths[i], 0);
        preprocess(img, maskImg);
        //train on images
        dt.train(img, depthImg, maskImg, true);
        //show result
        imshow("rgb", img);
        imshow("mask", maskImg);
        if(waitKey(5) == 32)
            break;
    }

    vector<String> outFilePrefix;
    outFilePrefix.push_back(configFilePrefix);
    outFilePrefix.push_back(featureFilePrefix);
    outFilePrefix.push_back(modelFilePrefix);

    //save trained models
    dt.save(outFilePrefix);
    return 0;
}

void imagePathLoader(const char *fileName, vector<String> &list, int n) {
    ifstream fs;
    fs.open(fileName);

    string str;
    int i=0;
    while(i < n) {
        fs >> str;
        list.push_back(str);
        i++;
    }
}

// prepocess images right now only resizing is done
void preprocess(Mat &img, Mat &mask) {
    resize(img, img, Size(640,360));
    resize(mask, mask, Size(640,360));
}


