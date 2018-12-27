//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "KFilter.h"

using namespace std;
using namespace cv;

bool swappingDetection(Point2d &observation1, Point2d &observation2, Point2d previousPosition1, Point2d previousPosition2);

//Error correction methods
int detectionMissingTuning(Point2d &observation1, Point2d &observation2, KFilter &kF1, KFilter &kF2, int frameCount, Mat &mouse1, Mat &mouse2);
void deviatedPositionTuning(Point2d &observation1, Point2d &observation2, KFilter &kF1, KFilter &kF2, int frameCount, Mat &mouse1, Mat &mouse2);

//Data Association Method
//Hungarian
int HungarianMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, KFilter kF1, KFilter kF2);
//Greedy
int GreedyMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, KFilter kF1, KFilter kF2);

void drawOnFrame(int asso, int frameCount, Mat mouse2, Mat mouse1, Point2d observation1, Point2d observation2, KFilter &kF1, KFilter &kF2, Mat &src);
#pragma once
