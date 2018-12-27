//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Eigen/Dense"

using namespace std;
using namespace cv;
using namespace Eigen;

/***** Object Detection *****/
Matrix2d getMice(Mat flow, Mat src, Mat flowthres);
void drawMice(Mat &image, Mat &src, Scalar boxColor);
bool findMiceByColor(Mat &src, Mat &src_gray, Mat &thres_output, Matrix2d &centroid_s);
Matrix2d devideFromFlow(Mat flow, Mat &thres_output, Mat &src);

/***** Foraging Detection *****/
Matrix2d isForaging(Mat foodMask1, Mat foodMask2, Mat mouseMask1, Mat mouseMask2);

/*** Tools ****/
double getDistance(Point2d pointO, Point2d pointA);
Vector2d pointToVector(Point2d point);
//Put Text
void PutText(Mat &src, string text, Point origin);
Point2d findCentroid(Mat &image);
int imageAllBlack(Mat src);
#pragma once

