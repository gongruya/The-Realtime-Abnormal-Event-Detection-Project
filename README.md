The Realtime Abnormal Event Detection Project
=====
This project is focused on behavior analysis in a surveillance system. 
###Algorithms
The current version of code is written in combination of Objective-C and C++. We used OpenCV to rewrite the learning algorithm inspired by the Matlab code based on the following paper,
> [1] "[Abnormal Event Detection at 150 FPS in Matlab](http://appsrv.cse.cuhk.edu.hk/~cwlu/Anormality_1000_FPS/abnormal_final3.pdf)" , Cewu Lu, Jianping Shi, Jiaya Jia, in ICCV, 2013

###Dataset
The subway dataset is provided by Professor Amit Adam
[http://www.cs.technion.ac.il/~amita](http://www.cs.technion.ac.il/~amita/).

####Update Logs
1.0.0 alpha - The prototype of realtime anomaly detection. It reaches 90FPS for local video decoding and detection. The parameters and learning model are not optimized yet, and we did not carry out experiment on labeled dataset.