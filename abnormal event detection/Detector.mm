//
//  Detector.mm
//  abnormal event detection
//
//  Created by Gong Ruya on 9/17/14.
//  Copyright (c) 2014 Gong Ruya. All rights reserved.
//

#import "Detector.h"
#import "AppDelegate.h"
#import "publicMethod.h"

@implementation frameDiffQueue
- (id) init {
    if (self = [super init]) {
        count = 0;
    }
    return self;
}
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
    ++count;
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
- (void) extractFeatures4Testing: (frameDiffQueue *) frames : (feaParams *) featuringParameters : (cv::Mat) COEFF {
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
            cv::Mat cube;
            cube = img[0](rect).clone().reshape(0, 1);
            for (int k = 1; k < depth; ++k) {
                cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);    //convert to a row vector
            }

            if (cv::sum(cube)[0] >= motionThr) {
                features.push_back(cube.clone());
                locX.push_back(i);
                locY.push_back(j);
            }
        }
    if (!features.empty())
        features = features * COEFF;
}
- (void) extractFeatures4Training: (frameDiffQueue *) frames : (feaParams *) featuringParameters {
    UInt8 depth = featuringParameters -> depth;
    UInt8 winH = featuringParameters -> winH;
    UInt8 winW = featuringParameters -> winW;
    UInt8 ssr = featuringParameters -> ssr;
    UInt8 tsr = featuringParameters -> tsr;
    double motionThr = featuringParameters -> motionThr;
    
    if (frames -> count % tsr) return;
    
    std::vector<cv::Mat> img;
    for (int k = 0; k < depth; ++k) {
        img.push_back(frames -> Queue[k]);
    }
    
    for (int i = 0; i < 160 - winW; i += ssr)
        for (int j =0; j < 120 - winH; j += ssr) {
            cv::Rect rect(i, j, winW, winH);
            cv::Mat cube;
            cube = img[0](rect).clone().reshape(0, 1);
            //std::cout << img[0](rect) << endl << endl;
            for (int k = 1; k < depth; ++k)
                cv::hconcat(cube, img[k](rect).clone().reshape(0, 1), cube);

            //std::cout << cv::sum(cube)[0] << endl;
            if (cv::sum(cube)[0] >= motionThr)
                features.push_back(cube.clone());
        }
}
- (size_t) featureNum {
    return features.rows;
}
@end

@implementation feaParams
- (id) init {
    if (self = [super init]) {
        ssr = 5, tsr = 1, depth = 5;
        winH = winW = 10;
        winHNum = 12, winWNum = 16;
        motionThr = 5;
    }
    return self;
}
@end

@implementation learningParams
- (id) init {
    if (self = [super init]) {
        Dim = 20;
    }
    return self;
}
@end

@implementation testingParams
- (void) setThreshold: (double)threshold {
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
    UInt32 maxIter = 1500;
    double alpha = 3e-2;
    //double threshold = learningParameters -> thr;
    double threshold = 0.3;
    
    int maxRows = 50;
    //std::cout << "Features remain: " << feaMat.rows << std::endl;
    addLog(@"Features remain: %d\n", feaMat.rows);
    int rows4CurrentTurn = maxRows;//feaMat.rows;
    while (feaMat.rows) {
        ///S 150 by 20     beta 20 by 1
        //threshold = 0.0493 * log(7.5887 + R.size());
        if (rows4CurrentTurn < sparseDim)
            break;                  //Nothing we can do
        
        cv::Mat feaMatCurrentTurn = feaMat(cv::Range(0, rows4CurrentTurn), cv::Range::all());
        
        cv::Mat S = cv::Mat(N, sparseDim, CV_64FC1);
        if (rows4CurrentTurn >= sparseDim) {
            cv::Mat centers, labels, tmp;
            feaMatCurrentTurn.convertTo(tmp, CV_32FC1);
            cv::kmeans(tmp, sparseDim, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
            ((cv::Mat)centers.t()).convertTo(S, CV_64FC1);
        } else {
            cv::theRNG().state = time(0);
            randu(S, cv::Scalar::all(0), cv::Scalar::all(0.1));
        }
    
        std::vector<cv::Mat> beta(feaMatCurrentTurn.rows);
        for (int j = 0; j < beta.size(); ++j) {
            beta[j] = cv::Mat(sparseDim, 1, CV_64FC1);
            randu(beta[j], cv::Scalar::all(0), cv::Scalar::all(0.1));
        }
            
        for (UInt32 iter = 1; iter <= maxIter; ++iter) {
            ///TRAINING
            cv::Mat gradOfS = [self gradientOfS:S andBeta:beta andFeature:feaMatCurrentTurn andAlpha:alpha];
            //std::vector<cv::Mat> gradOfBeta = [self gradientOfBeta:beta andS:S andFeature:feaMatCurrentTurn andAlpha:alpha];
            //cout << gradOfBeta[2] << endl;
            //cout << gradOfS << endl;
            
            S -= gradOfS;
            //for (int j = 0; j < beta.size(); ++j)
            //    beta[j] -= gradOfBeta[j];
            cv::Mat tmp = (S.t() * S).inv(cv::DECOMP_SVD) * S.t();
            for (int j = 0; j < beta.size(); ++j) {
                beta[j] = tmp * feaMatCurrentTurn.row(j).t();
            }
            
            if (!(iter % 500)) {
                //addLog(@"Grad Beta: %lf, Grad S: %lf\n", cv::norm(gradOfBeta[0]), cv::norm(gradOfS));
                addLog(@"Iter %d: L(S, beta) = %lf\n", iter, [self val: beta: S: feaMatCurrentTurn]);
            }
        }
        //cv::normalize(S, S);
        cv::Mat curR = (S * (S.t() * S).inv(cv::DECOMP_SVD) * S.t() - cv::Mat::eye(S.rows, S.rows, CV_64FC1)).t();
        
        cv::Mat result = feaMat * curR; //M by 500
        cv::Mat feaRemain;
        BOOL works = NO;
        for (int j = 0; j < result.rows; ++j) {
            double error = ((cv::Mat)(result.row(j) * result.row(j).t())).at<double>(0, 0);
            if (error > threshold)
                feaRemain.push_back(feaMat.row(j));
            else
                works = YES;
        }
        if (works == YES) {
            feaMat = feaRemain.clone();
            feaRemain.release();
            R.push_back(curR);
            //rows4CurrentTurn = feaMat.rows;
            rows4CurrentTurn = min(maxRows, feaMat.rows);
        } else {
            rows4CurrentTurn >>= 1;         //Reduce the data scale for training
            feaMat = shuffleRows(feaRemain);
            //feaMat = feaRemain.clone();
            feaRemain.release();
            addLog(@"Reducing data scale to: %d\n", rows4CurrentTurn);
        }
        addLog(@"Features remain: %d\n", feaMat.rows);
        addLog(@"Total detectors: %d, Threshold: %lf\n", R.size(), threshold);
    }
    addLog(@"Total detectors: %d\n", R.size());
}

- (cv::Mat) gradientOfS:(cv::Mat)S andBeta:(std::vector<cv::Mat>)beta andFeature:(cv::Mat)fea andAlpha:(double)alpha {
    cv::Mat ans = cv::Mat::zeros(S.rows, S.cols, CV_64FC1);
    for (int j = 0; j < fea.rows; ++j)
        ans += 2 * alpha * (S * beta[j] - fea.row(j).t()) * beta[j].t();
    return ans;
}
- (std::vector<cv::Mat>) gradientOfBeta:(std::vector<cv::Mat>)beta andS:(cv::Mat)S andFeature:(cv::Mat)fea andAlpha:(double)alpha {
    UInt64 size = beta.size();
    std::vector<cv::Mat> ans(beta.size());
    for (int j = 0; j < size; ++j) {
        ans[j] = 2 * alpha * S.t() * (S * beta[j] - fea.row(j).t());
    }
    return ans;
}

- (double) val: (std::vector<cv::Mat>) beta: (cv::Mat) S: (cv::Mat) fea {
    double ans = 0;
    for (int j = 0; j < fea.rows; ++j) {
        double t = cv::norm(fea.row(j).t() - S * beta[j]);
        ans += t * t;
    }
    return ans;
}

- (void) saveToFile: (NSString *) fileName {
    ///feaDim(4B), sparseDim(4B), size(8B), Data(8B * feaDim * feaDim * number)
    FILE *fp = fopen([fileName UTF8String], "wb");
    fwrite(&feaDim, sizeof(UInt32), 1, fp);
    fwrite(&sparseDim, sizeof(UInt32), 1, fp);
    UInt64 size = R.size();
    fwrite(&size, sizeof(UInt64), 1, fp);
    for (int k = 0; k < size; ++k)
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j)
                fwrite(&R[k].at<double>(i, j), sizeof(double), 1, fp);
    fclose(fp);
}
- (void) initFromFile: (NSString *) fileName {
    FILE *fp = fopen([fileName UTF8String], "rb");
    fread(&feaDim, sizeof(UInt32), 1, fp);
    fread(&sparseDim, sizeof(UInt32), 1, fp);
    UInt64 size;
    fread(&size, sizeof(UInt64), 1, fp);
    double tmp;
    for (int k = 0; k < size; ++k) {
        R.push_back(cv::Mat::zeros(feaDim, feaDim, CV_64FC1));
        for (int i = 0; i < feaDim; ++i)
            for (int j = 0; j < feaDim; ++j) {
                fread(&tmp, sizeof(double), 1, fp);
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
