//
//  AppDelegate.h
//  abnormal event detection
//
//  Created by Gong Ruya on 9/16/14.
//  Copyright (c) 2014 ___FULLUSERNAME___. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#include<fstream>
#import "NSImage+OpenCV.h"
#import "Detector.h"
//#import "RTSPServer.h"

using namespace cv;
using namespace std;

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (weak) IBOutlet NSImageView *videoPlay;
@property (weak) IBOutlet NSImageView *videoGray;
@property (weak) IBOutlet NSImageView *videoDiff;
@property (weak) IBOutlet NSImageView *videoResult;
@property (weak) IBOutlet NSTextField *FPS;

@property (assign) IBOutlet NSWindow *window;
- (IBAction)actSelectVideo:(id)sender;
- (IBAction)actSparseLearning:(id)sender;

@end