//
//  AppDelegate.m
//  abnormal event detection
//
//  Created by Gong Ruya on 9/16/14.
//  Copyright (c) 2014 ___FULLUSERNAME___. All rights reserved.
//

#import "AppDelegate.h"
#include "publicMethod.h"
#include <asl.h>
#include<pthread.h>

@implementation AppDelegate
@synthesize videoDisplay1;
@synthesize videoDisplay2;
@synthesize videoDisplay3;
@synthesize videoDisplay4;
@synthesize FPS;
@synthesize frameLabel;
@synthesize myLog;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    ///Initialization
    cv::theRNG().state = time(0);
    ///Global variables initialization
    extern pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    extern NSTextView *theLogText;
    theLogText = myLog;

    ///Test code
    //for (int i = 0; i < 1000; ++i) addLog(@"%d\n", i);
    //NSLog(@"%@", [csl objectAtIndex:0]);
    //NSLog(@"%@", [RTSPServer getIPAddress]);
    //Mat A = Mat::eye(10,10,CV_64FC1);
    //cout << A(Range::all(), Range(4,10)) << endl;
    //cout << shuffleRows(A) << endl;
    //detector *D = [detector new];
    //[D initFromFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector"];
    
}

- (void) training: (NSString *) videoPath {
    VideoCapture capture([videoPath UTF8String]);
    addLog(@"%@\n", @"Initializing the sparse learning system...");
    cuboid *features = [cuboid new];
    learningParams *myLearningParameters = [learningParams new];
    myLearningParameters -> Dim = 20;
    myLearningParameters -> thr = 0.35;
    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    
    cv::Size videoSize(160, 120);

    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];

    detector *myDetector = [detector new];   //Learning a detector
    int maxFrames = 1000;
    addLog(@"%@\n", @"Starting feature extraction...");
    capture.set(CV_CAP_PROP_POS_FRAMES, 40800);
    for (UInt64 i = 40800; i <= 40800 + maxFrames; ++i) {
        Mat frame, gray;
        if (!capture.read(frame)) break;
        
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        char img_name[500];
        sprintf(img_name, "/Users/gongruya/v/%llu.jpg", i);
        imwrite(img_name, frame);/*
        GaussianBlur(frame, frame, cv::Size(5,5), 0, 0, BORDER_DEFAULT);
        cvtColor(frame, gray, CV_BGR2GRAY);
        cv::normalize(gray, gray, 0, 1, NORM_MINMAX, CV_64FC1);
        [theFrames addDiff: gray];
        if (i > myFeatureParameters -> depth)
            [features extractFeatures4Training: theFrames: myFeatureParameters];
        */
        if (!(i % 100)) {
            addLog(@"%llu, ", i);
        }
    }
    addLog(@"%@\n", @"Feature extraction of the training video is done.");
    addLog(@"rows:%d\n", features->features.rows);
    
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaMat": features -> features.clone()];
    
    Mat myFea = features -> features.clone();
    PCA pca(myFea, cv::Mat(), CV_PCA_DATA_AS_ROW, 150);
    /*
    features -> features = pca.project(myFea).clone();
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaPCA": features -> features.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvalues": pca.eigenvalues.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvectors": pca.eigenvectors.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAmean": pca.mean.clone()];
    */
    addLog(@"%@\n", @"PCA for the training video is done.");
    addLog(@"rows:%d, cols: %d\n", features->features.rows, features->features.cols);

    //myFea = features -> features.clone();
    //features->features = shuffleRows(myFea).clone();
    //addLog(@"%@\n", @"Rows shuffled.");
    
    [myDetector sparseLearning: features: myLearningParameters];
    addLog(@"%@\n", @"Sparse learning is done.");
    
    [myDetector saveToFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector"];
    addLog(@"%@\n", @"Detector has been saved.");

}

- (void) myDemo: (NSString *) videoPath {
    /// Video Processing
    NSDate *timeStart = [NSDate date];
    
    VideoCapture capture([videoPath UTF8String]);

    double rate = capture.get(CV_CAP_PROP_FPS);
    int totalFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    capture.set(CV_CAP_PROP_POS_FRAMES, 40800);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 1);
    
    addLog(@"Video FPS: %lf, Total Frames: %d\n", rate, totalFrames);
    
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];

    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    
    cv::Size videoSize(160, 120);
    
    testingParams *myTestingParameters = [testingParams new];
    [myTestingParameters setThreshold: 0.5];
    
    detector *myDetector = [detector new];   //Load the detector
    [myDetector initFromFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector"];

    vector<NSDate *> timer;
    timer.push_back([NSDate date]);

    //cv::VideoWriter writer("/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize);
    

    Mat COEFF = [self loadMat:@"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCA"];
    for (UInt64 i = 0; ; ++i) {
        Mat frame, gray, frameRGB;
        if (!capture.read(frame)) break;
        frameLabel.stringValue = [NSString stringWithFormat:@"%llu", i];
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        GaussianBlur(frame, frame, cv::Size(5,5), 0, 0, BORDER_DEFAULT);   ///Smooth
        cvtColor(frame, gray, CV_BGR2GRAY);

        [self showVideo: gray at: 2];
        cv::normalize(gray, gray, 0, 1, NORM_MINMAX, CV_64FC1);
        
        cvtColor(frame, frameRGB, CV_BGR2RGB);
        //[self showVideo: frameRGB at: 1];
        
        [theFrames addDiff: gray];          ///Add current frame into the queue and calculate diff
        if (i > myFeatureParameters -> depth) {
            cuboid *fea = [cuboid new];
            [fea extractFeatures4Testing: theFrames: myFeatureParameters: COEFF];
            
            detectResult *result = [detectResult new];
            [result detect: myDetector: fea: myTestingParameters];
            
            Mat mask = Mat::zeros(myFeatureParameters -> winHNum, myFeatureParameters -> winWNum, CV_8UC1);
            size_t totAbn = [result abnormalNum];
            for (int j = 0; j < totAbn; ++j)
                mask.at<UInt8>(result -> locY[j], result -> locX[j]) = 255;
            cv::resize(mask, mask, videoSize, 0, 0, INTER_NEAREST);
            [self showVideo: mask at: 4];
            mask.release();
        }

        //[NSThread sleepForTimeInterval: delay];
        //if (i > 1) [self showVideo: [theFrames last] at: 3];              //The 1st frame cannot get difference

        if (timer.size() == 30) {
            timer.erase(timer.begin());
            timer.push_back([NSDate date]);
            FPS.stringValue = [NSString stringWithFormat:@"%.f", -30 / [timer[0] timeIntervalSinceNow]];
        } else {
            timer.push_back([NSDate date]);
        }
    }
    addLog(@"%@\n", @"Done");
    addLog(@"FPS: %.2f\n", -totalFrames / [timeStart timeIntervalSinceNow]);

}

- (void) showVideo: (Mat) frame at: (NSInteger) type {
    Mat img;
    cv::resize(frame, img, cv::Size(320,240), 0, 0, INTER_CUBIC);
    switch (type) {
        case 1:
            [videoDisplay1 setImage:[NSImage imageWithCVMat: img]];
            break;
        case 2:
            [videoDisplay2 setImage:[NSImage imageWithCVMat: img]];
            break;
        case 3:
            [videoDisplay3 setImage:[NSImage imageWithCVMat: img]];
            break;
        case 4:
            [videoDisplay4 setImage:[NSImage imageWithCVMat: img]];
            break;
        default:
            break;
    }
   
}

- (IBAction)actSelectVideo:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    [panel setMessage:@""];
    [panel setPrompt:@"OK"];
    [panel setCanChooseDirectories:NO];
    [panel setCanCreateDirectories:YES];
    [panel setCanChooseFiles:YES];
    NSString *path_all;
    NSInteger result = [panel runModal];
    if (result == NSFileHandlingPanelOKButton) {
        path_all = [[panel URL] path];
        addLog(@"%@\n", path_all);
        [NSThread detachNewThreadSelector:@selector(myDemo:) toTarget:self withObject:path_all];
    }
}

- (IBAction)actSparseLearning:(id)sender {
    
    NSString *path_all = @"";
    addLog(@"%@\n", path_all);
    [NSThread detachNewThreadSelector:@selector(training:) toTarget:self withObject:path_all];
}

- (void)saveMat: (NSString *)fileName: (cv::Mat)matrix {
    ///rows(4B), cols(4B), Data(8B * rows * cols)
    FILE *fp = fopen([fileName UTF8String], "wb");
    fwrite(&matrix.rows, sizeof(int), 1, fp);
    fwrite(&matrix.cols, sizeof(int), 1, fp);

    for (int i = 0; i < matrix.rows; ++i)
        for (int j = 0; j < matrix.cols; ++j)
            fwrite(&matrix.at<double>(i, j), sizeof(double), 1, fp);
    fclose(fp);
}
- (cv::Mat)loadMat: (NSString *)fileName {
    ///rows(4B), cols(4B), Data(8B * rows * cols)
    int r, c;
    FILE *fp = fopen([fileName UTF8String], "rb");
    
    fread(&r, sizeof(int), 1, fp);
    fread(&c, sizeof(int), 1, fp);
    cv::Mat matrix = cv::Mat::zeros(r, c, CV_64FC1);
    
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            double tmp;
            fread(&tmp, sizeof(double), 1, fp);
            matrix.at<double>(i, j) = tmp;
        }
    fclose(fp);
    return matrix;
}
@end
