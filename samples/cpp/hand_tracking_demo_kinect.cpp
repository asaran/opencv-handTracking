/*
 *      Author: guru
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/contrib/hd_per_pix.hpp>

#include <sstream>
#include <string>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;

// State of the hand
struct state {
    Rect region;
};

// initialize PerPixRegression with desired parameters
void initializeDetector(HT::PerPixRegression &dt);
// Load and preprocess images
bool loadImage(Mat &img, Mat &depthImg, Mat &depthMap);
// select target from image
void userSelectsTarget(Mat &_inpImg, state &st);
// mouse callback used for userSelectsTarget
void mouse_callback(int event,int x,int y,int ,void* param);
// initialize tracker
void initializeTracker(Mat &img, Mat &mask);
// Track hand and output its states
bool track(Mat &img, state &s);
// match object
bool matchObject(Mat &img, state &s);
// Segment hand from depthImg to be used as positive labels for training
void segmentHand(Mat &depthImg, state &s, Mat &mask);
// Function to show images and mask used for training
void showImages(Mat &img, Mat &mask, state &st);
// Convert from probImg (output of detector) to state
bool convertToState(Mat probImg, state &st);
// Function to show output of tracker
void showOutput(Mat &img, state &st);

int main() {
    // Mat objects to store image, depth image and mask - extracted from depth image
    Mat img, depthImg, mask, depthMap, img_res, depth_res, mask_res;;

    // No of Images to train for
    int trainingImages = 10;

    //State of the object - initially given by user
    state st;

    /*--------------------------Training---------------------------------*/
    // Training method used - PerPixRegression
    HT::PerPixRegression dt;
    // Initialize Hand detector with some customs params
    initializeDetector(dt);

    loadImage(img, depthImg, depthMap);
    userSelectsTarget(img, st);
    segmentHand(depthMap, st, mask);
    //imshow("mask", mask);
    //waitKey(30);
    initializeTracker(img, mask);
    showImages(img, mask, st);

    for(int i=0; i<trainingImages; i++) {
        // Load and preprocess images
        loadImage(img, depthImg, depthMap);
        segmentHand(depthMap, st, mask);
        // track hand
        track(img, st);
        // Extract mask from depthImg - rightnow for PerPixRegression depth is
        // not used internally in training so only used for mask extraction (may
        // be used in future to add some features

        // show images/mask used for training
        showImages(img, mask, st);
        // train detector
        resize(img, img_res, Size(400, 400));
        resize(mask, mask_res, Size(400, 400));
        resize(depthMap, depth_res, Size(400, 400));
        dt.train(img_res, depth_res, mask_res, true);
    }
    // Optionally save the trained models
    // path to save the configuration, features, models
    String configPath = "/home/guru/train_kinect";
    String featurePath = "/home/guru/train_kinect/features";
    String modelPath = "/home/guru/train_kinect/models";
    // vector to pass to save method, paths must be ordered as it is
    vector<String> paths;
    paths.push_back(configPath);
    paths.push_back(featurePath);
    paths.push_back(modelPath);
    // save models
    dt.save(paths);

    /*------------------------------Detection---------------------------*/
    // Output image given by detector
    Mat probImg;
    // State of the tracked hand
    // No of frames the detector must try to detect hand - if it fails, it quits
    int tryingThreshold = 3;
    // flag indicating if tracking active;
    bool trackingFlag = true;

    //loadImage(img, depthImg, depthMap);
    //userSelectsTarget(img, st);
    //segmentHand(depthMap, st, mask);
    //initializeTracker(img, depthMap, mask, st);

    while(loadImage(img, depthImg, depthMap)) {
        resize(img, img_res, Size(200, 200));
        resize(mask, mask_res, Size(200, 200));
        resize(depthMap, depth_res, Size(200, 200));

        //dt.detect(img, depthMap, probImg);
        //imshow("out", probImg);
        //track(img, depthMap, st);
        //showOutput(img, st);
        // track hand - if not - detect hand using PerPixRegression
        if(!track(img, st)) {
            // detect hand
            dt.detect(img_res, depth_res, probImg);
            // convert probImg to state, if unsuccessful try for tryingThrehold frames
            for(int i=0; i<tryingThreshold; i++) {
                if(!convertToState(probImg, st)) {
                    if(!loadImage(img, depthImg, depthMap))
                        break;
                    dt.detect(img, depthMap, probImg);
                }
                else
                    break;

                if(i == tryingThreshold-1) {
                    trackingFlag = false;
                    break;
                }
            }
            showOutput(img, st);
        }
        else {
            // show output
            showOutput(img, st);
        }
        // if tracking inactive - quits
        if(trackingFlag == false) {
            cout << "tracking failure\n";
            break;
        }
    }
    return 0;
}

// initialize PerPixRegression with desired parameters
void initializeDetector(HT::PerPixRegression &dt) {
    // Parameters for Detector
    HT::PerPixRegression::Params params;
    /*
     * -- do some parameter changes rightnow using default
     */
    dt = HT::PerPixRegression();
}

// Load and preprocess images
string imagePath = "/home/guru/Dropbox/HandTrackingDataset/set6/rgb/";
string depthPath = "/home/guru/Dropbox/HandTrackingDataset/set6/depth/";
stringstream ss;
int maxImages = 100;

// decode depth image from CV_32FC3 image to CV_16U - 16-bit depth resolution (depth in mm)
void decodeDepthImage(Mat &encoded, Mat &decoded);

// Loads and process images
bool loadImage(Mat &img, Mat &depthImg, Mat &depthMap) {
    static int i = 0;
    i++;
    if(i <= maxImages) {
        char temp[4];
        sprintf(temp, "%03d", i);

        // apply proper naming
        ss.str("");
        ss << imagePath << "rgb" << temp << ".png";
        //cout << ss.str();
        // read image
        img = imread(ss.str());

        ss.str("");
        ss << depthPath << "depth" << temp << ".png";
        // read encoded depth image
        depthImg = imread(ss.str());

        // depth images are saved in coded format in CV_32FC3 images and need to be decoded
        // to get full 16-bit depth resolution
        decodeDepthImage(depthImg, depthMap);

        // preprocess imgaes
        //resize(img, img, Size(200, 200));
        //resize(depthImg, depthImg, Size(200, 200));
        //resize(depthMap, depthMap, Size(200, 200));
        return true;
    }
    else {
        return false;
    }

}

// decode depth image from CV_32FC3 image to CV_16U - 16-bit depth resolution (depth in mm)
void decodeDepthImage(Mat &encoded, Mat &decoded) {
    if(!decoded.data || encoded.rows != decoded.rows || encoded.cols != decoded.cols)
        decoded = Mat(encoded.rows, encoded.cols, CV_16U);
    for(int i=0; i<encoded.rows; i++) {
        for(int j=0; j<encoded.cols; j++) {
            decoded.at<ushort>(i,j) = (int(encoded.at<Vec3b>(i,j)[0])*256 + int(encoded.at<Vec3b>(i,j)[1]));
        }
    }
}

// Function to show images and mask used for training
void showImages(Mat &img, Mat &mask, state &st) {
    // create image for final output
    Mat showImg(img.rows, 2*img.cols, CV_8UC3, Scalar(0));
    Mat roi, roi1, roi2;
    // create region of interest for rgbImg
    roi = Mat(showImg, Rect(0, 0, img.cols, img.rows));
    // copy rgbImg to roi
    img.copyTo(roi);
    rectangle(roi, st.region, Scalar(255,0,0));
    //imshow("rgb image, depth image, extracted mask to train", showImg);
    //waitKey(0);

    //roi1 = Mat(showImg, Rect(img.cols, 0, img.cols, img.rows));
    //depthImg.convertTo(roi1, CV_8UC1);
    Mat mask_temp;
    cvtColor(mask, mask_temp, COLOR_GRAY2BGR);
    roi2 = Mat(showImg, Rect(img.cols, 0, img.cols, img.rows));
    mask_temp.copyTo(roi2);
    rectangle(roi2, st.region, Scalar(255,0,0));
    //imshow("rgb image, depth image, extracted mask to train", showImg);
    //waitKey(0);

    //imshow("image", img);
    //imshow("mask", mask);

    imshow("rgb image, depth image, extracted mask to train", showImg);
    waitKey(30);
}

bool selected, drawing_box;
Rect box;

//function to select the hand region using mouse by the user
void userSelectsTarget(Mat &_inpImg, state &st) {
    //temperory object used for this function
    Mat temp;
    //selected - boolean showing if the a valid rectangular region has been selected for the target
    //drawing_box - boolean showing what current rectangular area is selected
    selected = drawing_box = false;

    cout << "Drag the mouse on the image for selecting the area. Try to select a tight bounding rect\n";

    //window for the image
    namedWindow("Select Target");
    //mouse callback used for selection
    setMouseCallback("Select Target",mouse_callback,(void*) &temp);

    while(selected == false) {
        _inpImg.copyTo(temp);
        if(drawing_box == true) {
            //draw rectangle on the image if a valid rectangle has been made
            rectangle(temp, box, Scalar(0, 255, 0), 1);
        }
        //output the image
        imshow("Select Target", temp);
        char c = waitKey(5);
        if(c == 32)
            break;
    }
    destroyWindow("Select Target");
    st.region = box;
}

//mouse callback function to select the target region
void mouse_callback(int event,int x,int y,int ,void* param) {
    Mat *image = (Mat*) param;
    switch( event ){
        case EVENT_MOUSEMOVE:
            if( drawing_box ){
                box.width = x-box.x;
                box.height = y-box.y;
            }
            break;

        case EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = cv::Rect( x, y, 0, 0 );
            break;

        case EVENT_LBUTTONUP:
            drawing_box = false;
            if( box.width < 0 ){
                box.x += box.width;
                box.width *= -1;
            }
            if( box.height < 0 ){
                box.y += box.height;
                box.height *= -1;
            }
            cv::rectangle(*image,box,cv::Scalar(0,255,0),1);
            selected = true;
            break;
    }

}

Mat hist[4];
Mat backPro[4];
float def_hranges[] = {0, 180};
float def_sranges[] = {0, 256};
float def_vranges[] = {0, 256};
const int noOfBins[3] = {30, 32, 32};
const float *ranges[] = {def_hranges, def_sranges, def_vranges};
int ch[] = {0, 1, 2};

// object model
MatND objHist;

TermCriteria criteria;
//initialize tracker
void initializeTracker(Mat &img, Mat &mask) {
    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    vector<Mat> channel;
    split(img_hsv, channel);
    calcHist(&channel[0], 1, 0, mask, hist[0], 1, &noOfBins[0], ranges);
    calcHist(&channel[1], 1, 0, mask, hist[1], 1, &noOfBins[1], ranges+1);
    calcHist(&channel[2], 1, 0, mask, hist[2], 1, &noOfBins[2], ranges+2);

    //calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    calcHist(&img_hsv, 1, ch, mask, objHist, 2, noOfBins, ranges);
    normalize(objHist, objHist, 0, 1, NORM_MINMAX, -1, Mat() );
}
// track hand
bool track(Mat &img, state &s) {
    Mat imgCpy, binOut;
    cvtColor(img, imgCpy, COLOR_BGR2HSV);

    vector<Mat> channel;
    split(imgCpy, channel);
    calcBackProject(&channel[0], 1, 0, hist[0], backPro[0], ranges);
    calcBackProject(&channel[1], 1, 0, hist[1], backPro[1], ranges+1);
    calcBackProject(&channel[2], 1, 0, hist[2], backPro[2], ranges+2);

    multiply(backPro[0], backPro[1], binOut, 1./255.0, CV_32F);
    multiply(backPro[2], binOut, binOut, 1./255.0, CV_32F);

    meanShift(binOut, s.region, criteria);
    return matchObject(img, s);
}

MatND tempHist;
double threshCompare = 0.7;

//match object using histogram matching
bool matchObject(Mat &img, state &s) {
    Mat imgCpy;
    cvtColor(img, imgCpy, COLOR_BGR2HSV);
    Mat roi = imgCpy(s.region);
    calcHist(&roi, 1, ch, Mat(), tempHist, 2, noOfBins, ranges);
    normalize(tempHist, tempHist, 0, 1, NORM_MINMAX, -1, Mat());
    double val = compareHist(objHist, tempHist, HISTCMP_BHATTACHARYYA);
    bool flag;
    if(val < threshCompare)
        flag =  true;
    else
        flag = false;
    return flag;
}


// Segment hand region from depth image

const unsigned char EMPTY = 0;
const unsigned char HAND = 255;
const unsigned int SENSOR_MIN = 100;
const unsigned int SENSOR_MAX = 7000;

queue<pair<int, int> > _pixels;
int _depthThr = 40;
int _maxObjectSize = 1000000;

void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth);
pair<int, int> searchNearestPixel(const Mat &depth, Rect &region);

// segment hand region from the depth
void segmentHand(cv::Mat &depth, state &s, Mat &mask) {
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(depth.type() == CV_16UC1);

    if(mask.data) {
        CV_Assert(mask.rows == depth.rows);
        CV_Assert(mask.cols == depth.cols);
    }
    else
        mask = Mat(depth.rows, depth.cols, CV_8UC1);

    mask.setTo(EMPTY);

    pair<int, int> current = searchNearestPixel(depth, s.region);
    if (current.first < 0){
        return;
    }

    int rowcount = s.region.height, colcount = s.region.width;
    Mat visited(depth.rows, depth.cols, CV_8U, Scalar(0));


    double mean = depth.at<unsigned short>(current.first,current.second);
    int minx=depth.cols,miny=depth.rows,maxx=0,maxy=0;

    int pixelcount = 1;
    _pixels.push(current);

    while((!_pixels.empty()) & (pixelcount < _maxObjectSize))
    {
        current = _pixels.front();
        _pixels.pop();

        if (current.first < minx) minx = current.first;
                else if (current.first > maxx) maxx = current.first;
        if (current.second < miny) miny = current.second;
                else if (current.second > maxy) maxy = current.second;

        if ( current.first + 1 < rowcount+s.region.y && visited.at<uchar>(current.first+1, current.second) == 0 ){
            visited.at<uchar>(current.first+1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first + 1,current.second,depth);
        }

        if ( current.first - 1 > -1 + s.region.y && visited.at<uchar>(current.first-1, current.second) == 0){
            visited.at<uchar>(current.first-1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first - 1,current.second,depth);
        }

        if ( current.second + 1 < colcount + s.region.x && visited.at<uchar>(current.first, current.second+1) == 0){
            visited.at<uchar>(current.first, current.second+1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second + 1,depth);
        }

        if( current.second - 1 > -1 + s.region.x && visited.at<uchar>(current.first, current.second-1) == 0){
            visited.at<uchar>(current.first, current.second-1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second - 1,depth);
        }
    }
}

// search nearest pixel in the selected region
pair<int, int> searchNearestPixel(const Mat &depth, Rect &region) {
    pair<int, int> pt;
    pt.first = -1;
    pt.second = -1;
    const unsigned short *depthptr;
    unsigned short min = (1<<15);
    for(int i=region.y; i<region.y+region.height; i++) {
        depthptr = depth.ptr<const unsigned short>(i);
        for(int j=region.x; j<region.x+region.width; j++) {
            if(depthptr[j] > SENSOR_MIN && depthptr[j] < SENSOR_MAX && depthptr[j] < min) {
                min = depthptr[j];
                pt.first = i;
                pt.second = j;
            }
        }
    }
    return pt;
}

// Add neighbouring pixel of an already labelled point
void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth) {
    unsigned short d = depth.at<unsigned short>(first,second );

    if ( mask.at<uchar>(first,second ) == EMPTY &&
         fabs(d-mean/pixelcount) < _depthThr && d > SENSOR_MIN && d <= SENSOR_MAX)
    {
        pixelcount++;
        mean += d;
        mask.at<uchar>(first,second ) = HAND;
        _pixels.push(pair<int, int>(first,second));
    }
}

int canny_thresh = 100;
int thresh_area = 200;
// convert probabilty image to state
bool convertToState(Mat probImg, state &st) {
    
    
    Mat grayImg;
    blur(probImg, grayImg, Size(3, 3));

    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( grayImg, canny_output, canny_thresh, canny_thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    int max_index = 0;
    double max_area = 0;
    for(int i=0; i<(int)contours.size(); i++) {
        double area = contourArea(contours[i]);
        if(area > max_area){
            max_area = area;
            max_index = i;
        }
    }

    if(max_area > thresh_area) {
        st.region = boundingRect(contours[max_index]);
        return true;
    }
    return false;
}

// Function to show output of tracker
void showOutput(Mat &img, state &st) {
    // create image for final output
    Mat showImg(img.rows, img.cols, CV_8UC3);
    // copy image to output image
    img.copyTo(showImg);

    // draw state to show
    rectangle(showImg, st.region, Scalar(255,0,0), 1);
    // show output image
    imshow("tracked output", showImg);
    waitKey(100);
}


