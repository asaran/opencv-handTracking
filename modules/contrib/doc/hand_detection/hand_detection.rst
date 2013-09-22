Hand Detection
==============

.. highlight:: cpp

HandDetector
------------

.. ocv:class:: HandDetector

Base class providing common interface to various Hand Detection methods namely : 
 * HistBackProj    
 * PerPixRegression


Declaration::

    class CV_EXPORTS HandDetector {
    
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
    
HandDetector::train 
-------------------
Trains the detector on a single image and return true on successful training. 

.. ocv:function:: bool HandDetector::train(Mat & _rgbImg, Mat & _depthImg, Mat & _mask, bool incremental)

    :param _rgbImg: The input RGB/color image with CV_8UC3 type.
    
    :param _depthImg: The input depth image with CV_16UC1 type. Whether to use depth information depends on 'useDepth' parameter in individual methods.
    
    :param _mask: Mask image showing valid hand region/patch with CV_8UC1 type
     
    :param incremental: Passed in as true if we want to append the new data for training. If false, all previous learned models are discarded and training starts fresh.


HandDetector::detect 
--------------------
Detects whether there is hand or not in the image and outputs an probability image

.. ocv:function:: void HandDetector::detect(Mat & _rgbImg, Mat & _depthImg, OutputArray probImg) 

    :param _rgbImg: The input color image on which to detect - CV_8UC3
    
    :param _depthImg: The input _depthImg providing additional information used for detection (may not be used at all, depends on individual detection method) - CV_16UC1.
    
    :param _probImg: The output probability image - CV_8UC1

        
HandDetector::load
------------------
Loads a learned classifier from a file and returns true if successful 

.. ocv:function:: bool HandDetector::load(vector<String> &fileNamePrefix)
 
    :param fileNamePrefix: Path to the folder. Paths should be specified in the order {configPath, featurePath, modelPath} where
        
        * configPath - path for configuration file (config.xml)
        * featurePath - path for saving HSV features extracted (hsv0.xml, hsv1.xml ...)
        * modelPath - path for saving models (model0.xml,model1.xml)
        

HandDetector::save
------------------

Saves a learned classifier to a file and returns true if successful 

.. ocv:function:: bool HandDetector::save(vector<String> &fileNamePrefix)

    :param fileNamePrefix: The path to the folder where the learned models are saved. For formatting refer load() function above.
    

