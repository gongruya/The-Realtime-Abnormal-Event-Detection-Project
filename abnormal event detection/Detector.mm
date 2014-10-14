//
//  Detector.mm
//  abnormal event detection
//
//  Created by Gong Ruya on 9/17/14.
//  Copyright (c) 2014 Gong Ruya. All rights reserved.
//

#import "Detector.h"

@implementation frameDiffQueue
- (void) add: (cv::Mat) obj {
    if (Queue.size() == size) {
        Queue.erase(Queue.begin());
        Queue.push_back(obj);
    } else {
        Queue.push_back(obj);
    }
}
- (void) addDiff: (cv::Mat) frame {
    cv::Mat diff;
    if (Queue.size() > 0)
        cv::absdiff(frame, current, diff);
    current = frame;
    [self add: diff];
    diff.release();
}
- (cv::Mat) last {
    return Queue.back();
}
- (cv::Mat) orig {
    return current;
}
- (void) setSize: (UInt8) sz {
    size = sz;
}
- (std::vector<cv::Mat>) val {
    return Queue;
}
- (UInt8) length {
    return size;
}
@end

@implementation cuboid
- (void) extractFeatures4Testing: (frameDiffQueue *) frames : (feaParams *) featuringParameters {
    UInt8 depth = featuringParameters -> depth;
    UInt8 winH = featuringParameters -> winH;
    UInt8 winW = featuringParameters -> winW;
    UInt8 winHNum = featuringParameters -> winHNum;
    UInt8 winWNum = featuringParameters -> winWNum;
    double motionThr = featuringParameters -> motionThr;
    
    std::vector<cv::Mat> img;
    for (int k = 0; k < depth; ++k) {
        img.push_back(frames -> Queue[k]);
    }
    
    for (int i = 0; i < winWNum; ++i)
        for (int j = 0; j < winHNum; ++j) {
            cv::Rect rect(i * winW, j * winH, winW, winH);
            //NSLog(@"%d %d %d %d",i * winW, j * winH, winW, winH);
            cv::Mat cube;
            cube = img[0](rect).clone().reshape(0, 1);
            for (int k = 1; k < depth; ++k) {
                cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);    //convert to a row vector
            }
            cube.convertTo(cube, CV_64FC1);
            cube /= 255;
            //std::cout << cv::sum(cube)[0];
            if (cv::sum(cube)[0] >= motionThr) {
                features.push_back(cube.clone());
                locX.push_back(i);
                locY.push_back(j);
            }
        }
}
- (void) extractFeatures4Training: (frameDiffQueue *) frames : (feaParams *) featuringParameters {
    UInt8 depth = featuringParameters -> depth;
    UInt8 winH = featuringParameters -> winH;
    UInt8 winW = featuringParameters -> winW;
    UInt8 ssr = featuringParameters -> ssr;
    double motionThr = featuringParameters -> motionThr;
    std::vector<cv::Mat> img;
    for (int k = 0; k < depth; ++k) {
        img.push_back(frames -> Queue[k]);
    }
    
    for (int i = 0; i < 160 - winW; i += ssr)
        for (int j =0; j < 120 -winH; j += ssr) {
            cv::Rect rect(i, j, winW, winH);
            cv::Mat cube;
            cube = img[0](rect).clone().reshape(0, 1);
            for (int k = 1; k < depth; ++k)
                cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);
            cube.convertTo(cube, CV_64FC1);
            cube /= 255;
            //std::cout << cv::sum(cube)[0];
            if (cv::sum(cube)[0] >= motionThr)
                features.push_back(cube.clone());
        }
    
}
- (size_t) featureNum {
    return features.rows;
}
@end

@implementation feaParams
@end

@implementation learningParams
@end

@implementation testingParams
- (void) setParams: (double)threshold {
    thr = threshold;
}
@end

@implementation detector
- (size_t) detectorNum {
    return R.size();
}
- (void) sparseLearning:(cuboid *)features :(learningParams *)learningParameters {
    cv::Mat feaMat = features -> features;
    UInt32 N = feaMat.cols;
    sparseDim = learningParameters -> Dim;
    feaDim = N;
    UInt32 maxIter = 500;
    double alpha = 0.03;
    double threshold = learningParameters -> thr;
    cv::theRNG().state = time(0);           //Random seed
    std::cout << "Features remain: " << feaMat.rows << std::endl;
    while (feaMat.rows) {
        ///S 500 by 20     beta 20 by 1
        cv::Mat S = cv::Mat(N, sparseDim, CV_64FC1);
        randu(S, cv::Scalar::all(0), cv::Scalar::all(0.1));
        std::vector<cv::Mat> beta(feaMat.rows);
        for (UInt32 iter = 1; iter <= maxIter; ++iter) {
            ///TRAINING
            cv::Mat tmp = (S.t() * S).inv(cv::DECOMP_SVD) * S.t();
            for (int j = 0; j < feaMat.rows; ++j)
                beta[j] = tmp * feaMat.row(j).t();
            cv::Mat grad = [self gradient: beta: S: feaMat];
            S -= alpha * grad;
            if (!(iter % 10))
                std::cout << "Iter " << iter << ": " << cv::norm(grad) << std::endl;
        }
        cv::normalize(S, S);
        cv::Mat curR = (S * (S.t() * S).inv(cv::DECOMP_SVD) * S.t() - cv::Mat::eye(S.rows, S.rows, CV_64FC1)).t();

        cv::Mat result = feaMat * curR; //M by 500
        cv::Mat feaRemain;
        BOOL works = NO;
        for (int j = 0; j < result.rows; ++j) {
            cv::Mat error = result.row(j) * result.row(j).t();
            if (error.at<double>(0, 0) > threshold)
                feaRemain.push_back(feaMat.row(j));
            else
                works = YES;
        }
        if (works == YES) {
            feaMat = feaRemain.clone();
            feaRemain.release();
            R.push_back(curR);
        } else
            break;
        std::cout << "Features remain: " << feaMat.rows << std::endl;
    }
}

- (cv::Mat) gradient: (std::vector<cv::Mat>) beta: (cv::Mat) S: (cv::Mat) fea {
    cv::Mat ans = cv::Mat::zeros(S.rows, S.cols, CV_64FC1);
    for (int j = 0; j < fea.rows; ++j)
        ans += 2 * (S * beta[j] - fea.row(j).t()) * beta[j].t();
    return ans;
}

- (void) saveToFile: (NSString *) fileName {
    ///feaDim(4B), sparseDim(4B), size(8B), Data(8B * feaDim * feaDim * number)
    FILE *fp = fopen([fileName UTF8String], "wb");
    fwrite(&feaDim, 4, 1, fp);
    fwrite(&sparseDim, 4, 1, fp);
    UInt64 size = R.size();
    fwrite(&size, 8, 1, fp);
    for (int k = 0; k < size; ++k)
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j)
                fwrite(&R[k].at<double>(i, j), 8, 1, fp);
    fclose(fp);
}
- (void) initFromFile: (NSString *) fileName {
    FILE *fp = fopen([fileName UTF8String], "rb");
    fread(&feaDim, 4, 1, fp);
    fread(&sparseDim, 4, 1, fp);
    UInt64 size;
    fread(&size, 8, 1, fp);
    double tmp;
    for (int k = 0; k < size; ++k) {
        R.push_back(cv::Mat::zeros(feaDim, feaDim, CV_64FC1));
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j) {
                fread(&tmp, 8, 1, fp);
                R[k].at<double>(i, j) = tmp;
            }
    }
    fclose(fp);
}
@end

@implementation detectResult
- (void) detect:(detector *)myDetector :(cuboid *)cuboidFeature :(testingParams *)testingParameters {
    size_t feaNum = [cuboidFeature featureNum];
    size_t detNum = [myDetector detectorNum];
    double thr = testingParameters -> thr;
    for (int i = 0; i < feaNum; ++i) {
        normal.push_back(NO);
        for (int j = 0; j < detNum; ++j) {
            cv::Mat error = cuboidFeature -> features.row(i) * myDetector -> R[j];
            error *= error.t();
            //std::cout << error.at<double>(0,0) << std::endl;
            if (error.at<double>(0,0) < thr) {
                normal[i] = YES;
                break;
            }
        }
        if (normal[i] == NO) {
            locX.push_back(cuboidFeature -> locX[i]);
            locY.push_back(cuboidFeature -> locY[i]);
        }
    }
}
- (size_t) abnormalNum {
    return locX.size();
}
@end
