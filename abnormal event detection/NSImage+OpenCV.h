//
//  NSImage+OpenCV.h
//

#import <AppKit/AppKit.h>

@interface NSImage (NSImage_OpenCV) {
    
}

+(NSImage*)imageWithCVMat:(const cv::Mat&)cvMat;
-(id)initWithCVMat:(const cv::Mat&)cvMat;

@property(nonatomic, readonly) cv::Mat CVMat;
@property(nonatomic, readonly) cv::Mat CVGrayscaleMat;

@end