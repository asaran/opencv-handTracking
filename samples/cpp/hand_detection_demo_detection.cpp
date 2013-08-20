#include <opencv2/contrib/hand_detector.hpp>
#include <opencv2/contrib/hd_per_pix.hpp>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;

// function to convert a text file of image paths to a vector of paths
void imagePathLoader(const char *fileName, vector<String> &list, int n);
// function to perform general preprocessing steps
void preprocess(Mat &src, Mat &dest);

int main() {

    string imgNameFilePath = "/home/guru/trained2/imgFile.txt";
    // list of image and mask file paths (with file type)
    vector<String> imgPaths;
    // no of images to train for
    int noOfImages = 10;
    // load image paths
    imagePathLoader(imgNameFilePath.c_str(), imgPaths, noOfImages);

    // path to configuration file - trained parameters
    String configFilePrefix = "/home/guru/trained2/config";
    // path to trained feature files
    String featureFilePrefix = "/home/guru/trained2/features/feature";
    //path to trained model files
    String modelFilePrefix = "/home/guru/trained2/models/model";

    vector<String> outFilePrefix;
    outFilePrefix.push_back(configFilePrefix);
    outFilePrefix.push_back(featureFilePrefix);
    outFilePrefix.push_back(modelFilePrefix);

    ////////////////// Main sample codes starts ///////////////////////

    HT::PerPixRegression dt;

    // load trained models
    dt.load(outFilePrefix);

    // Mat object for bgr images
    Mat img, im;
    // Mat object for Dummy depth image - PerPixRegression doesnot uses depth
    Mat depth = Mat();
    // Mat object for output
    Mat probImg;
    for(int i=0; i<10; i++) {
        img = imread(imgPaths[i]);
        preprocess(img, im);

        dt.detect(im, depth, probImg);

        imshow("source", img);
        imshow("probImg", probImg);
        if(waitKey(0) == 32)
            break;
    }
}

VideoCapture cap;

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

void preprocess(Mat &src, Mat &dest) {
    Mat _src;
    src.copyTo(_src);
    /*MatIterator_<Vec3b> it = _src.begin<Vec3b>();
    MatIterator_<Vec3b> end = _src.end<Vec3b>();
    for( ; it != end; it++) {
        Vec3b& pix = *it;
        int temp = pix[0];
        pix[0] = pix[2];
        pix[2] = temp;
    }*/
    resize(src, dest, Size(200, 200));
}

