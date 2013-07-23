#include <opencv2/contrib/hd_color_model.hpp>
#include <opencv2/contrib/hand_detector.hpp>
#include <opencv2/nonfree.hpp>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <queue>
#include <cmath>

#include <iostream>

using namespace std;
using namespace cv;

const unsigned char EMPTY = 0;
const unsigned char HAND = 255;
const unsigned short SENSOR_MIN = 400;
const unsigned short SENSOR_MAX = 7000;

void userSelectsTarget(Mat &_inpImg);
void segmentHand(cv::Mat &mask, Rect &region, const cv::Mat &depth);
pair<int, int> searchNearestPixel(const Mat &depth, Rect &region);
void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth);
void mouse_callback(int event,int x,int y,int ,void* param);

class CV_EXPORTS gpCapture {
public:
    VideoCapture cap;
    VideoCapture cap1;
    Mat rgbImg;
    Mat depthImg, depthMap;
    int useVideo;

    gpCapture(void) {
        useVideo = -1;
    }

    gpCapture(int h) {
        cap.open(h);
        if(h == CAP_OPENNI)
            useVideo = 0;
        else
            useVideo = 1;
    }

    gpCapture(string h) {
        const string rgbN = h+"/rgb/rgb%03d.png";
        const string depthN = h+"/depth/depth%03d.png";
        cap.open(rgbN);
        cap1.open(depthN);
        useVideo = 2;
        depthMap = Mat(480, 640, CV_16UC1);
    }

    bool update(void) {
        bool result = false;
        if(useVideo == 0) {
            result = cap.grab();
            cap.retrieve(rgbImg, CAP_OPENNI_BGR_IMAGE);
            cap.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP);
        }
        else if(useVideo == 1) {
            result = cap.grab();
            cap.retrieve(rgbImg);
            depthMap = Mat();
        }
        else if(useVideo == 2) {
            result = (cap.grab() && cap1.grab());
            if(result) {
                cap.retrieve(rgbImg);
                cap1.retrieve(depthImg);
                for(int i=0; i<depthImg.rows; i++) {
                    for(int j=0; j<depthImg.cols; j++) {
                        depthMap.at<ushort>(i,j) = (int(depthImg.at<Vec3b>(i,j)[0])*256 + int(depthImg.at<Vec3b>(i,j)[1]));
                    }
                }
            }
        }
        return result;
    }
} capture;


const String keys =
        "{help h usage ? |      | print this message   }"
        "{kinect         |0     | use kinect           }"
        "{@path          |.     | path to dataset      }"
        "{color          |1     | use color            }"
        "{depth          |0     | use depth            }"
        ;

//Rectangular box used user selection of the target region in the RGB image
Rect box;

int main(int argc, char *argv[]) {

    //options evaluated from the command line arguments
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    //specifies if the kinect (RGBD camera device) is to be used
    bool use_kinect = false;
    use_kinect = parser.get<int>("kinect");

    //specifies if depth have to be used
    bool use_depth = false;
    use_depth = parser.get<int>("depth");

    //specifies if color info is to be used
    bool use_color = false;
    use_color = parser.get<int>("color");

    //opens up a VideoCapture object according to the options
    if(use_kinect == true)
        capture = gpCapture(CAP_OPENNI);
    else {
        string datasetPath = parser.get<string>(0);
        capture = gpCapture(datasetPath);
    }

    //check if openning of the device is successful
    if(!capture.cap.isOpened()) {
        cout << "cannot open input stream\n";
        return 0;
    }

    //windows to display output images
    namedWindow("Input - Output");

    //update capture to get input images
    capture.update();

    //lets user select a target - the hand region using mouse in RGB Image
    userSelectsTarget(capture.rgbImg);

    //Mask calculated for the input image using corresponding depth image
    Mat mask = Mat();
    if(!capture.depthMap.empty()) {
        mask = Mat(capture.depthMap.rows, capture.depthMap.cols, CV_8UC1);
        segmentHand(mask, box, capture.depthMap);
    }
    else {
        cout << "cannot calculate mask using depth";
    }

    //vector specifying no. of bins for the color histograms
    vector<int> noOfBins(4);
    noOfBins[0] = 30;
    noOfBins[1] = 32;
    noOfBins[2] = 32;
    //extra number used to define no. of bins for depth histogram
    noOfBins[3] = 256;

    //HistBackProj object used for color-based histogram backprojection method
    //HT::HistBackProj dt;
    HT::HistBackProj::Params params;
    params.useColor = use_color;
    params.useDepth = use_depth;
    Ptr<HT::HandDetector> dt = new HT::HistBackProj(params);
    //train dt with initial input images
    dt->train(capture.rgbImg, capture.depthMap, mask, false);

    //probability output image
    Mat probImg, probImgRGB;
    int rows = capture.rgbImg.rows, cols = capture.rgbImg.cols;
    Mat OutImg = Mat(rows, 2*cols, CV_8UC3, Scalar(0));
    Mat temp1 = OutImg(Rect(0,0,cols, rows));
    Mat temp2 = OutImg(Rect(cols,0,cols,rows));

    //Main loop where detection is taking place
    while(capture.update()) {
        //detect is the main function which performs detection
        dt->detect(capture.rgbImg, capture.depthMap, probImg);

        //display input rgb image and output probability image
        capture.rgbImg.copyTo(temp1);
        cvtColor(probImg, probImgRGB, COLOR_GRAY2BGR);
        probImgRGB.copyTo(temp2);
        imshow("Input - Output", OutImg);
        if(waitKey(1) == 32)
            break;
    }

    //destroy Output windows
    destroyWindow("Input - Output");
    return 0;
}

//some variables used for selection of the target region (hand region) by the user
bool selected, drawing_box;

//function to select the hand region using mouse by the user
void userSelectsTarget(Mat &_inpImg) {
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
}

//_pixels - used to hold individual pixel which are to be used in region growing for segmenting hand in depth image
queue<pair<int, int> > _pixels;
//_depthThr - used to specify bound determining whether to add a pixel in the mask
int _depthThr = 40;
//_maxObjectSize - specifies the max. no. of pixel the mask can have
int _maxObjectSize = 10000;

//function used to create a mask using the depth image for the selected target
void segmentHand(cv::Mat &mask, Rect &region, const cv::Mat &depth)
{
    //Assert statements to check compatibility among various inputs
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(depth.type() == CV_16UC1);

    CV_Assert(mask.rows == depth.rows);
    CV_Assert(mask.cols == depth.cols);

    //mask set to EMPTY - no hand region
    mask.setTo(EMPTY);
    //nearest pixel in the selected region is used to seed the region growing of mask
    pair<int, int> current = searchNearestPixel(depth, region);
    if (current.first < 0){
        return;
    }

    int rowcount = region.height, colcount = region.width;
    //specifies whether the pixel has been already visited in the region growing
    Mat visited(depth.rows, depth.cols, CV_8U, Scalar(0));
    //used to accumulate depth - thus mean/pixelCount gives exact 'mean'
    double mean = depth.at<unsigned short>(current.first,current.second);
    //pixelcount - represents total number of hand pixel added to the mask
    int pixelcount = 1;

    _pixels.push(current);

    while((!_pixels.empty()) & (pixelcount < _maxObjectSize))
    {
        current = _pixels.front();
        _pixels.pop();

        //check all the neighbouring pixels of the current pixel and process it if is valid and not visited before
        if ( current.first + 1 < rowcount+region.y && visited.at<uchar>(current.first+1, current.second) == 0 ){
            visited.at<uchar>(current.first+1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first + 1,current.second,depth);
        }

        if ( current.first - 1 > -1 + region.y && visited.at<uchar>(current.first-1, current.second) == 0){
            visited.at<uchar>(current.first-1, current.second) = 255;
            processNeighbor(pixelcount,mean,mask,current.first - 1,current.second,depth);
        }

        if ( current.second + 1 < colcount + region.x && visited.at<uchar>(current.first, current.second+1) == 0){
            visited.at<uchar>(current.first, current.second+1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second + 1,depth);
        }

        if( current.second - 1 > -1 + region.x && visited.at<uchar>(current.first, current.second-1) == 0){
            visited.at<uchar>(current.first, current.second-1) = 255;
            processNeighbor(pixelcount,mean,mask,current.first,current.second - 1,depth);
        }
    }
}

//function to search nearest pixel in depth image in the rectangular area specified by 'region'
pair<int, int> searchNearestPixel(const Mat &depth, Rect &region) {
    pair<int, int> pt;
    pt.first = -1;
    pt.second = -1;
    const unsigned short *depthptr;
    unsigned short min = (1<<15);
    for(int i=region.y; i<region.y+region.height; i++) {
        depthptr = depth.ptr<const unsigned short>(i);
        for(int j=region.x; j<region.x+region.width; j++) {
            //check if the pixel is in sensor range
            if(depthptr[j] > SENSOR_MIN && depthptr[j] < SENSOR_MAX && depthptr[j] < min) {
                min = depthptr[j];
                pt.first = i;
                pt.second = j;
            }
        }
    }
    return pt;
}

//function that process a pixel and determines if it is to be added as hand region in the mask
void processNeighbor(int &pixelcount, double &mean, cv::Mat &mask, const short first, const short second, const cv::Mat &depth)
{
    unsigned short d = depth.at<unsigned short>(first,second );

    //if the current location in mask is not already hand region, is within sensor range and
    //difference between its distance 'd' and 'mean' (=mean/pixelcount) is smaller than threshold
    if ( mask.at<uchar>(first,second ) == EMPTY &&
         fabs((double)d-(double)mean/(double)pixelcount) < _depthThr && d > SENSOR_MIN && d <= SENSOR_MAX)
    {
        pixelcount++;
        mean += d;
        //HAND represents true value for hand region
        mask.at<uchar>(first,second ) = HAND;
        _pixels.push(pair<int, int>(first,second));
    }
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

