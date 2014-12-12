//
//  publicMethod.m
//  Realtime Abnormal Event Detection
//
//  Created by Gong Ruya on 10/23/14.
//  Copyright (c) 2014 Gong Ruya. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "publicMethod.h"

NSTextView *theLogText;
pthread_mutex_t lock;

void *addLogThread(void *arguments) {
    NSString *str = [NSString stringWithUTF8String: (char *)arguments];
    NSLog(@"%@", str);
    return NULL;
}
void addLog(NSString *formatStr, ...) {
    //pthread_mutex_lock(&lock);
    va_list arglist;
    va_start(arglist, formatStr);
    NSString *str =[[NSString alloc] initWithFormat:formatStr arguments:arglist];
    /*
    const char *s = [str UTF8String];
    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_create(&tid, &attr, &addLogThread, (void *)s);
    pthread_join(tid, NULL);
    */
    //pthread_mutex_unlock(&lock);
    //NSAttributedString *strAttr =[[NSAttributedString alloc] initWithString: str];
    //[[theLogText textStorage] beginEditing];
    //[[[theLogText textStorage] mutableString] appendString: str];
    /*[[theLogText textStorage] performSelectorOnMainThread:@selector(appendAttributedString:)
                                           withObject:str
                                        waitUntilDone:YES];
    */
    //[[theLogText textStorage] endEditing];
    //[theLogText scrollRangeToVisible:NSMakeRange([[theLogText string] length], 0)];
    NSLog(@"%@", str);
}


cv::Mat shuffleRows(const cv::Mat &matrix) {
    std::vector <int> seeds;
    for (int i = 0; i < matrix.rows; ++i)
        seeds.push_back(i);
    cv::theRNG().state = time(0);
    cv::randShuffle(seeds);
    cv::Mat output;
    for (int i = 0; i < matrix.rows; ++i)
        output.push_back(matrix.row(seeds[i]));
    return output;
}