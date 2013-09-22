#include <iostream>
#include <opencv2/contrib/hd_per_pix.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// loads images and masks - user specific for naming convention
void loadImage(Mat &img, Mat &mask, string imagePath, string maskPath, string extension);
// shows output
void showOutput(Mat &img, Mat &mask, String winName);

static void help()
{
    cout << "\n-----------------------------------------------------------------------\n"
            "This is demo shows how to use PerPixRegression class for hand detection\n"
            "First the detector is trained on some images and then testing is done on\n"
            " a video. The demo optionally lets you save trained models.\n"
            "------------------------------------------------------------------------\n"
            "\nUsage: \n"
            "   ./perPixRegressionDemo [params]\n"
            "params : \n"
            "[--imgPath /path to training images/frame]\n"
            "[--maskPath /mask to training masks/mask]\n"
            "[--extension .jpg(extension of images)]\n"
            "[--testVideo /path to test video/...avi]\n"
            "[--savePath /path to save models/ (not saved if not specified)]\n"
            "[--trainingImages 5(no of training images)]\n";

    cout << "\n\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tSPACE - pause/resume the program\n\n";
}

const char* keys =
{
    "{help h usage ?|     | print this message}"
    "{imgPath       |     | path to training images}"
    "{maskPath      |     | path to mask images}"
    "{extension     | .jpg| extension of the images}"
    "{testVideo     |     | path to test video}"
    "{savePath      |     | path to save models}"
    "{trainingImages|  5  | no of training images}"
};

int main(int argc, const char** argv) {
    help();

    string imagePath = "/home/guru/versionControl/gsocWorkspace/database/gp/frames/frame";
    string maskPath = "/home/guru/versionControl/gsocWorkspace/database/gp/masks/mask";
    string extension = ".jpg";

    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if(parser.has("imgPath") && parser.has("maskPath")) {
        imagePath = parser.get<string>("imgPath");
        maskPath = parser.get<string>("maskPath");
    }
    else {
        cout << "\nUsing default paths for images\n";
    }
    cout << "Image path : " << imagePath << "\n"
         << "Mask path : " << maskPath << "\n";

    if(parser.has("extension"))
        extension = parser.get<string>("extension");
    else
        cout << "\nUsing default extension\n";
    cout << "extension : "<< extension << "\n";


    // Mat objects to store image and mask - extracted from depth image
    Mat img, mask, img_res, mask_res, depth_res;
    
    // No of Images to train for
    int trainingImages = 5;
    if(parser.has("trainingImages"))
        trainingImages = parser.get<int>("trainingImages");

    /*--------------------------Training---------------------------------*/
    HT::PerPixRegression::Params params;
    //defines how many pixels are to skiped during training or detection step
    params.training_step_size = 5;
    params.testing_step_size = 3;

    // Training method used - PerPixRegression
    HT::PerPixRegression dt(params);

    namedWindow("training");
    for(int i=0; i<trainingImages; i++) {
        // Load and preprocess images
        loadImage(img, mask, imagePath, maskPath, extension);

        // show images/mask used for training
        showOutput(img, mask, "training");
        char c = waitKey(30);
        if(c == 27)
            return 0;
        if(c == 32) {
            c = waitKey(30);
            while(c != 32 && c != 27)
                c = waitKey(30);
            if(c == 27)
                return 0;
        }

        // train detector
        resize(img, img_res, Size(640, 480));
        resize(mask, mask_res, Size(640, 480));
        dt.train(img_res, depth_res, mask_res, true);
    }
    destroyWindow("training");

    // Optionally save the trained models
    if(parser.has("savePath")) {
        String configPath, featurePath, modelPath;
        configPath = featurePath = modelPath = parser.get<string>("savePath");

        // vector to pass to save method, paths must be ordered as it is
        vector<String> paths;
        paths.push_back(configPath);
        paths.push_back(featurePath);
        paths.push_back(modelPath);
        // save models
        dt.save(paths);
    }

    /*----------------------------Testing----------------------------*/
    // output probability image - CV-8UC1
    Mat probImg, probImg_res;

    //Dummy depth image - not used in perPixRegression for detection
    Mat depthImg = Mat();

    // video capture object to capture test video
    string testVideo = "/home/guru/Dropbox/_EDSH/vid/EDSH1.avi";
    if(parser.has("testVideo"))
        testVideo = parser.get<string>("testVideo");
    else
        cout << "\nUsing default testVideo\n";
       
    cout << "testVideo : " << testVideo << "\n";

    VideoCapture cap(testVideo);

    namedWindow("testing");
    while(cap.read(img)) {
        resize(img, img_res, Size(200, 200));
        // detect hand in image
        dt.detect(img_res, depthImg, probImg_res);
        // show output
        resize(img, img, Size(640, 480));
        resize(probImg_res, probImg, Size(640, 480));
        showOutput(img, probImg, "testing");
        char c = waitKey(5);
        if(c == 27)
            break;
        if(c == 32) {
            c = waitKey(30);
            while(c != 32 && c != 27)
                c = waitKey(30);
            if(c == 27)
                return 0;
        }
    }
    destroyWindow("testing");
    return 0;
}

// load and preprocess images

stringstream ss;

// Loads and process images - user specific
void loadImage(Mat &img, Mat &mask, string imagePath, string maskPath, string extension) {
    static int i = 0;
    i++;

    // apply proper naming
    ss.str("");
    ss << imagePath << i << extension;
    img = imread(ss.str());

    ss.str("");
    ss << maskPath  << i << extension;
    mask = imread(ss.str(), 0);
}

// show output
void showOutput(Mat &img, Mat &mask, String winName) {
    // output image with img and mask side-by-side
    Mat out(img.rows, 2*img.cols, CV_8UC3, Scalar(0));
    // create roi to copy img
    Mat roi = out(Rect(0, 0, img.cols, img.rows));
    img.copyTo(roi);
    // create roi to copy mask
    roi = out(Rect(img.cols, 0, img.cols, img.rows));
    cvtColor(mask, roi, COLOR_GRAY2BGR);
    //show images
    imshow(winName, out);
}
