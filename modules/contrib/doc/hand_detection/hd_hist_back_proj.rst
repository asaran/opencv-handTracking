Hand Detection
==============

.. highlight:: cpp

HandDetector::HistBackProj
-------------------------_

Derived class from hand detector class. Performs detection using Histogram matching through backProjected images.

Declaration::

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

HistBackProj::Params
--------------------

General structure defining various parameters for the HistBackProj detection.

    * **noOfBins[4]** - defines no. of bins for histograms for each channel – BGRD
    
        * default – {30, 32, 32, 0} i.e depth is not used.
    
    * **histRange[4][2]** - defines range of values for constructing histogram for each channel

        * default – {{0, 180}, {0, 256}, {0, 256}, {400, 7000}}.
        
    * **frameSize** - defines size of the frame, set when training.
    
    * **colorCode** - defines colorCode to use for changing among color spaces – codes to be used in cvtColor().

        * default COLOR_BGR2HSV
        
    * **useDepth** - defines whether to use depth or not.
    
        * default – false
        
    * **useColor** - defines whether to use color or not.
    
        * default - true

    .. ocv:function:: void read( const FileStorage& fn )
    
        reads param values from a file
        
    .. ocv:function:: void write( FileStorage& fs ) const
    
        saves param values to a file

