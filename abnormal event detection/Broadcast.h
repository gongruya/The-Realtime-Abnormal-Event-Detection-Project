//
//  Broadcast.h
//  Realtime Abnormal Event Detection
//
//  Created by Gong Ruya on 9/24/14.
//  Copyright (c) 2014 Gong Ruya. All rights reserved.
//

@interface broadcast: NSObject {
    
}
- (void) startDaemon;
- (void) sendFrame: (id) frame;
- (void) stopDaemon;
@end