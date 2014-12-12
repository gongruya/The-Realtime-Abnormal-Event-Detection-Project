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


using namespace cv;
using namespace std;

@interface AppDelegate : NSObject <NSApplicationDelegate>

@property (weak) IBOutlet NSTextField *frameLabel;
@property (weak) IBOutlet NSTextField *FPS;
@property (weak) IBOutlet NSImageView *videoDisplay1;
@property (weak) IBOutlet NSImageView *videoDisplay2;
@property (weak) IBOutlet NSImageView *videoDisplay3;
@property (weak) IBOutlet NSImageView *videoDisplay4;

@property (assign) IBOutlet NSWindow *window;
- (IBAction)actSelectVideo:(id)sender;
- (IBAction)actSparseLearning:(id)sender;
@property (unsafe_unretained) IBOutlet NSTextView *myLog;

@end