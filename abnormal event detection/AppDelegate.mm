//
//  AppDelegate.m
//  abnormal event detection
//
//  Created by Gong Ruya on 9/16/14.
//  Copyright (c) 2014 ___FULLUSERNAME___. All rights reserved.
//

#import "AppDelegate.h"

@implementation AppDelegate
@synthesize videoPlay;
@synthesize videoDiff;
@synthesize videoGray;
@synthesize videoResult;
@synthesize FPS;
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    //My code
    //NSLog(@"%@", [RTSPServer getIPAddress]);
}

- (void) training: (NSString *) videoPath {
    VideoCapture capture([videoPath UTF8String]);
    cuboid *features = [cuboid new];
    learningParams *myLearningParameters = [learningParams new];
    myLearningParameters -> Dim = 20;
    myLearningParameters -> thr = 0.8;
    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    myFeatureParameters -> winH = 10;
    myFeatureParameters -> winW = 10;
    myFeatureParameters -> winHNum = 12;
    myFeatureParameters -> winWNum = 16;
    myFeatureParameters -> ssr = 3;
    myFeatureParameters -> tsr = 2;
    myFeatureParameters -> motionThr = 5.0;
    myFeatureParameters -> depth = 5;
    cv::Size videoSize(160, 120);
    
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];
    
    detector *myDetector = [detector new];   //Learning a detector
    
    int maxFrames = 7500;
    for (UInt64 i = 0; i < maxFrames; ++i) {
        Mat frame, gray;
        if (!capture.read(frame)) break;
        
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        GaussianBlur(frame, frame, cv::Size(3,3), 0, 0, BORDER_DEFAULT);
        cvtColor(frame, gray, CV_BGR2GRAY);
        [theFrames addDiff: gray];
        if (i > myFeatureParameters -> depth)
            [features extractFeatures4Testing: theFrames: myFeatureParameters];
    }
    NSLog(@"%@\n", @"Feature extraction of the training video is done.");
    [myDetector sparseLearning: features: myLearningParameters];
    NSLog(@"%@\n", @"Sparse learning is done.");
    [myDetector saveToFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myDetector"];
    NSLog(@"%@\n", @"Detector has been saved.");
}

- (void) myDemo: (NSString *) videoPath {
    /// Video Processing
    NSDate *timeStart = [NSDate date];
    
    VideoCapture capture([videoPath UTF8String]);

    double rate = capture.get(CV_CAP_PROP_FPS);
    int totalFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 5000);
    cout << rate <<" "<< totalFrames << endl;
    
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];
    
    
    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    myFeatureParameters -> winH = 10;
    myFeatureParameters -> winW = 10;
    myFeatureParameters -> winHNum = 12;
    myFeatureParameters -> winWNum = 16;
    myFeatureParameters -> motionThr = 5.0;
    myFeatureParameters -> depth = 5;
    cv::Size videoSize(160, 120);
    
    testingParams *myTestingParameters = [testingParams new];
    [myTestingParameters setParams: 1];
    
    
    detector *myDetector = [detector new];   //Load the detector
    [myDetector initFromFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myDetector"];

    vector<NSDate *> timer;
    timer.push_back([NSDate date]);

    for (UInt64 i = 0; ; ++i) {
        Mat frame, gray, frameRGB;
        if (!capture.read(frame)) break;
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        GaussianBlur(frame, frame, cv::Size(3,3), 0, 0, BORDER_DEFAULT);   ///Smooth
        cvtColor(frame, gray, CV_BGR2GRAY);
        
        cvtColor(frame, frameRGB, CV_BGR2RGB);
        [self showVideo: frameRGB: 1.2];
        
        //[self showVideo: gray: 2];
        [theFrames addDiff: gray];          ///Add current frame into the queue and calculate diff
        if (i > myFeatureParameters -> depth) {
            cuboid *fea = [cuboid new];
            [fea extractFeatures4Testing: theFrames: myFeatureParameters];
            
            detectResult *result = [detectResult new];
            [result detect: myDetector: fea: myTestingParameters];
            
            Mat mask = Mat::zeros(myFeatureParameters -> winHNum, myFeatureParameters -> winWNum, CV_8UC1);
            size_t totAbn = [result abnormalNum];
            for (int j = 0; j < totAbn; ++j)
                mask.at<UInt8>(result -> locY[j], result -> locX[j]) = 255;
            cv::resize(mask, mask, videoSize, 0, 0, INTER_NEAREST);
            [self showVideo: mask: 4];
            mask.release();
        }

        //[NSThread sleepForTimeInterval: delay];
        if (i > 1) [self showVideo: [theFrames last]: 3];              //The 1st frame cannot get difference

        if (timer.size() == 30) {
            timer.erase(timer.begin());
            timer.push_back([NSDate date]);
            FPS.stringValue = [NSString stringWithFormat:@"%.f", -30 / [timer[0] timeIntervalSinceNow]];
        } else {
            timer.push_back([NSDate date]);
        }
    }
    NSLog(@"Done");
    NSLog(@"FPS: %.2f", -totalFrames / [timeStart timeIntervalSinceNow]);
}

- (void) showVideo: (Mat) frame: (NSInteger) type {
    switch (type) {
        case 1:
            [videoPlay setImage:[NSImage imageWithCVMat: frame]];
            break;
        case 2:
            [videoGray setImage:[NSImage imageWithCVMat: frame]];
            break;
        case 3:
            [videoDiff setImage:[NSImage imageWithCVMat: frame]];
            break;
        case 4:
            [videoResult setImage:[NSImage imageWithCVMat: frame]];
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
        NSLog(@"%@", path_all);
        [NSThread detachNewThreadSelector:@selector(myDemo:) toTarget:self withObject:path_all];
    }
}

- (IBAction)actSparseLearning:(id)sender {
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
        NSLog(@"%@", path_all);
        [NSThread detachNewThreadSelector:@selector(training:) toTarget:self withObject:path_all];
    }
}

@end
