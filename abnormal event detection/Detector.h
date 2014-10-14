//
//  Detector.h
//  abnormal event detection
//
//  Created by Gong Ruya on 9/17/14.
//  Copyright (c) 2014 Gong Ruya. All rights reserved.
//

///Detector
@interface learningParams: NSObject {           //Learning parameters
@public
    double thr, lambda;
    UInt32 Dim;
}
@end

@interface testingParams: NSObject {            //Testing parameters
@public
    double thr;
}
- (void) setParams: (double) threshold;         //To be continued
@end

@interface feaParams: NSObject {                //Feature extraction parameters
@public
    UInt8 ssr, tsr, depth;                      //Sliding window sample rate, time sample rate, cuboid depth
    UInt8 winH, winW, winHNum, winWNum;         //Window height, window width, number H, number W
    double motionThr;                           //Motion threshold
}
@end

@interface frameDiffQueue: NSObject {           //Nearest n frames
@public
    UInt8 size;                                 //Max size of the queue
    cv::Mat current;                            //Current frame and the
    std::vector<cv::Mat> Queue;
}
- (void) add: (cv::Mat) obj;
- (void) addDiff: (cv::Mat) frame;
- (cv::Mat) last;
- (cv::Mat) orig;
- (void) setSize: (UInt8) sz;
- (std::vector<cv::Mat>) val;
- (UInt8) length;
@end

@interface cuboid: NSObject {                   //A single spatio-temporal cuboid or a group of cuboids.
@public
    cv::Mat features;                           //Each row is a feature
    std::vector<UInt8> locX, locY;
}
- (void) extractFeatures4Testing: (frameDiffQueue *) frames : (feaParams *) featuringParameters;
- (void) extractFeatures4Training: (frameDiffQueue *) frames : (feaParams *) featuringParameters;
- (size_t) featureNum;
@end

@interface detector: NSObject {
@public
    std::vector<cv::Mat> R;
    UInt32 sparseDim;
    UInt32 feaDim;
}
- (void) sparseLearning: (cuboid *) features : (learningParams*) learningParameters;   //Train with cuboids and parameters.
- (size_t) detectorNum;
- (cv::Mat) gradient: (std::vector<cv::Mat>) beta : (cv::Mat) S: (cv::Mat) fea;
- (void) saveToFile: (NSString *) fileName;
- (void) initFromFile: (NSString *) fileName;
@end

@interface detectResult: NSObject {
@public
    std::vector<UInt8> locX, locY;
    std::vector<BOOL> normal;
}
- (void) detect: (detector *) myDetector : (cuboid *) cuboidFeature : (testingParams*) testingParameters;
- (size_t) abnormalNum;
@end
