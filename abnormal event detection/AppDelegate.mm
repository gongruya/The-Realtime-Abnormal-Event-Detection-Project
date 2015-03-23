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
    
    extern int detectorArea[9];
    
    cv::Mat dArea = cv::Mat::zeros(9, 16, CV_8UC1);
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 16; ++j)
            dArea.at<unsigned char>(i, j) = ((detectorArea[i] >> j) & 1) * 255;
    cv::resize(dArea, dArea, cv::Size(320, 180), 0, 0, INTER_NEAREST);
    [self showVideo:dArea at: 1];
}

- (void) training: (NSString *) videoPath {
    VideoCapture capture([videoPath UTF8String]);
    addLog(@"%@\n", @"Initializing the sparse learning system...");
    
    learningParams *myLearningParameters = [learningParams new];
    myLearningParameters -> thr = 0.1;
    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    
    
    cv::Size videoSize(160, 90);

    ///We train detector for each spatio patch
    
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];
    vector <detector *> detectorGroup(16*9);
    vector <cuboid *> features(16*9);
    for (int i = 0; i < 16*9; ++i) {
        detectorGroup[i] = [detector new];
        features[i] = [cuboid new];
    }
    
    if (1) {                            //Extracting??
        int maxFrames = 10375;
        addLog(@"%@\n", @"Starting feature extraction...");
        for (UInt64 i = 1; i <= maxFrames; ++i) {
            Mat frame, gray;
            if (!capture.read(frame)) break;
            cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
            GaussianBlur(frame, frame, cv::Size(3,3), 0, 0, BORDER_DEFAULT);
            cvtColor(frame, gray, CV_BGR2GRAY);
            cv::normalize(gray, gray, 0, 1, NORM_MINMAX, CV_64FC1);
            [theFrames addDiff: gray];
            if (i > myFeatureParameters -> depth) {
                for (int ii = 0; ii < 16; ++ii)
                    for (int jj = 0; jj < 9; ++jj) {
                        [features[jj*16+ii] extractFeatures4Training: theFrames: myFeatureParameters: ii: jj];
                    }
            }
            if (!(i % 100)) {
                addLog(@"%llu, ", i);
            }
        }
        addLog(@"%@\n", @"Feature extraction of the training video is done.");
        for (int i = 0; i < 16*9; ++i) {
            addLog(@"%d:%d\n", i, features[i]->features.rows);
            [self saveMat: [NSString stringWithFormat:@"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaMat%.3d", i]: features[i] -> features];
        }
    } else {
        addLog(@"%@", @"Loading...\n");
        for (int i = 0; i < 16*9; ++i) {
            features[i] -> features = [self loadMat: [NSString stringWithFormat:@"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaMat%.3d", i]];
        }
    }
    
    for (int i = 0; i < 16*9; ++i) {
        addLog(@"Start sparse learning for patch %d, rows: %d\n", i, features[i] -> features.rows);
        [detectorGroup[i] sparseLearning: features[i]: myLearningParameters];
        addLog(@"%@\n", @"Sparse learning is done.");
        [detectorGroup[i] saveToFile: [NSString stringWithFormat:@"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector%.3d", i]];
    }
    
    /*
    cuboid *features = [cuboid new];
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];

    detector *myDetector = [detector new];   //Learning a detector
    int maxFrames = 4000;
    addLog(@"%@\n", @"Starting feature extraction...");
    //capture.set(CV_CAP_PROP_POS_FRAMES, 40800);
    for (UInt64 i = 1; i <= maxFrames; ++i) {
        Mat frame, gray;
        if (!capture.read(frame)) break;
        
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        GaussianBlur(frame, frame, cv::Size(3,3), 0, 0, BORDER_DEFAULT);
        cvtColor(frame, gray, CV_BGR2GRAY);
        cv::normalize(gray, gray, 0, 1, NORM_MINMAX, CV_64FC1);
        [theFrames addDiff: gray];
        if (i > myFeatureParameters -> depth)
            [features extractFeatures4Training: theFrames: myFeatureParameters];
        if (!(i % 100)) {
            addLog(@"%llu, ", i);
        }
    }
    addLog(@"%@\n", @"Feature extraction of the training video is done.");
    addLog(@"rows:%d\n", features->features.rows);
    
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaMat": features -> features.clone()];
    
    Mat myFea = features -> features.clone();
    PCA pca(myFea, cv::Mat(), CV_PCA_DATA_AS_ROW, 150);
    
    features -> features = pca.project(myFea);
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/feaPCA": features -> features.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvalues": pca.eigenvalues.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvectors": pca.eigenvectors.clone()];
    [self saveMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAmean": pca.mean.clone()];
    
    addLog(@"%@\n", @"PCA for the training video is done.");
    addLog(@"rows:%d, cols: %d\n", features->features.rows, features->features.cols);

    //myFea = features -> features.clone();
    //features->features = shuffleRows(myFea).clone();
    //addLog(@"%@\n", @"Rows shuffled.");
    
    [myDetector sparseLearning: features: myLearningParameters];
    addLog(@"%@\n", @"Sparse learning is done.");
    
    [myDetector saveToFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector"];
    addLog(@"%@\n", @"Detector has been saved.");
    */
}

- (void) myDemo: (NSString *) videoPath {
    /// Video Processing
    NSDate *timeStart = [NSDate date];
    
    VideoCapture capture([videoPath UTF8String]);

    double rate = capture.get(CV_CAP_PROP_FPS);
    int totalFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 8186*5-250+9000);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 13540*5-250);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 3000);
    //capture.set(CV_CAP_PROP_POS_FRAMES, 17*60*25-250);
    capture.set(CV_CAP_PROP_POS_FRAMES, 3000);
    
    addLog(@"Video FPS: %lf, Total Frames: %d\n", rate, totalFrames);
    
    frameDiffQueue *theFrames = [frameDiffQueue new];
    [theFrames setSize: 5];

    ///set featuring parameters
    feaParams *myFeatureParameters = [feaParams new];
    
    cv::Size videoSize(160, 90);
    
    testingParams *myTestingParameters = [testingParams new];
    [myTestingParameters setThreshold: 0.2];
    double optThr = 0.2;
    
    /*detector *myDetector = [detector new];   //Load the detector
    [myDetector initFromFile: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector"];
    PCA pca;
    pca.eigenvalues = [self loadMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvalues"];
    pca.eigenvectors = [self loadMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAeigenvectors"];
    pca.mean = [self loadMat: @"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/PCAmean"];
    */
    
    
    vector <detector *> detectorGroup(16*9);
    for (int i = 0; i < 16*9; ++i) {
        detectorGroup[i] = [detector new];
        [detectorGroup[i] initFromFile: [NSString stringWithFormat:@"/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/myDetector%.3d", i]];
        addLog(@"Loading Detector %d\n", i+1);
    }
    
    vector<NSDate *> timer;
    timer.push_back([NSDate date]);

    //cv::VideoWriter writer("/Users/gongruya/Documents/Computer Vision/abnormal event detection/myData/VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize);
    int delayTime = 1;
    for (UInt64 i = 0; ; ++i) {
        Mat frame, gray, frameRGB;
        if (!capture.read(frame)) break;
        frameLabel.stringValue = [NSString stringWithFormat:@"%llu", i];
        cv::resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
        GaussianBlur(frame, frame, cv::Size(3,3), 0, 0, BORDER_DEFAULT);   ///Smooth
        cvtColor(frame, gray, CV_BGR2GRAY);

        [self showVideo: gray at: 2];
        cv::normalize(gray, gray, 0, 1, NORM_MINMAX, CV_64FC1);
        
        cvtColor(frame, frameRGB, CV_BGR2RGB);
        //[self showVideo: frameRGB at: 1];
        
        [theFrames addDiff: gray];          ///Add current frame into the queue and calculate diff
        if (i > myFeatureParameters -> depth) {
            cuboid *fea = [cuboid new];
            [fea extractFeatures4Testing: theFrames: myFeatureParameters];
            
            detectResult *result = [detectResult new];
            [result detect: detectorGroup: fea: myTestingParameters];
            
            /*Mat mask = Mat::zeros(myFeatureParameters -> winHNum, myFeatureParameters -> winWNum, CV_8UC1);
            size_t totAbn = [result abnormalNum];
            for (int j = 0; j < totAbn; ++j)
                mask.at<UInt8>(result -> locY[j], result -> locX[j]) = 255;
            */
            
            Mat mask, mask255;
            mask = result -> anomalyMap;
            
            GaussianBlur(mask, mask, cv::Size(3,3), 0, 0, BORDER_DEFAULT);
            
            mask255 = mask * 255;
            mask255.convertTo(mask255, CV_8UC1);
            threshold(mask255, mask255, optThr * 255, 255, CV_THRESH_BINARY);
            
            mask255.convertTo(mask, CV_64FC1);
            mask /= 255;
            
            cv::resize(mask, mask, videoSize, 0, 0, INTER_NEAREST);
            
            cv::Mat grayWithMask;
            cv::normalize(min(mask*0.7 + gray, 1), grayWithMask, 0, 255, NORM_MINMAX, CV_8UC1);
            
            [self showVideo: grayWithMask at:4];
            cv::resize(mask255, mask255, videoSize, 0, 0, INTER_NEAREST);
            [self showVideo:mask255 at:3];
            
            mask.release();
            mask255.release();
        }

        if (timer.size() == 30) {
            timer.erase(timer.begin());
            timer.push_back([NSDate date]);
            double fps = -30 / [timer[0] timeIntervalSinceNow];
            double tpf = 1/fps*1000;
            delayTime = max(1000/25 - (int)tpf, 1);
            FPS.stringValue = [NSString stringWithFormat:@"%.f", fps];
        } else {
            timer.push_back([NSDate date]);
        }
        //waitKey(delayTime);
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
        [NSThread detachNewThreadSelector:@selector(training:) toTarget:self withObject:path_all];
    }
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
