//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

//C++ standard libraries
#include <io.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <fstream>

//Eigen Library
#include "Eigen/Dense"

//Custom Library
#include "utility.h"
#include "KFilter.h"
#include "Tracking.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const static int startFrame = 7
;
const static double deltaT = (double) 1 / 30;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, Mat& zero, int step)
{
	float threshld = 0.1;
	for (int y = 0; y < cflowmap.rows; y += 1)
		for (int x = 0; x < cflowmap.cols; x += 1)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			if (abs(fxy.x) > threshld || abs(fxy.y) > threshld) {
				float param = fxy.y / fxy.x;
				zero.at<uchar>(y, x) = 255;
			}
		}
}

int main(int argc, char *argv[])
{
	VideoWriter writer("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(960, 540));
	KFilter kF1(Scalar(255, 0, 0));  //Blue
	KFilter kF2(Scalar(0, 255, 0));  //Green

	/******* Initial Starting Point of Kalman Filter ******/
	double kF11 = 466.278;
	double kF12 = 409.279;
	double kF21 = 611.344;
	double kF22 = 145.079;

	if (argc == 5)
	{
		sscanf_s(argv[1], "%lf", &kF11);
		sscanf_s(argv[2], "%lf", &kF12);
		sscanf_s(argv[3], "%lf", &kF21);
		sscanf_s(argv[4], "%lf", &kF22);
		cout << kF11 << endl;
		cout << kF12 << endl;
		cout << kF21 << endl;
		cout << kF22 << endl;
	}
	else if (argc != 1)
	{
		cout << "Only Accept Four Parameters!" << endl;
	}

	kF1.setInitialState(kF11, kF12);
	kF2.setInitialState(kF21, kF22);

	/*****************************************************/

	VideoCapture capture("./input.avi");
	
	if (!capture.isOpened())
	{
		printf("can not open ...\n");
		return -1;
	}

	Mat flow, cflow, frame;
	UMat gray, prevgray, uflow;

	/**** counters of time ****/
	double kF1Food1 = 0;
	double kF1Food2 = 0;
	double kF2Food1 = 0;
	double kF2Food2 = 0;
	/**************************/

	Mat foodMask1 = imread("Food1.png", IMREAD_COLOR);
	Mat foodMask2 = imread("Food2.png", IMREAD_COLOR);

	int frameCount = 0;

	while (1)
	{
		capture >> frame;
		if (frame.empty())
		{
			cout << "Frame is empty!" << endl;
			break;
		}
		resize(frame, frame, Size(800, 640));

		Matrix2d centroids;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		blur(gray, gray, Size(3, 3));
		blur(gray, gray, Size(3, 3));
		Mat zero = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		if (!prevgray.empty())
		{
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			uflow.copyTo(flow);
			drawOptFlowMap(flow, cflow, zero, 5);
			centroids = getMice(flow, frame, zero);
			//imshow("flowFilter", zero);
		}

		cout << "Frame: " << frameCount << endl;

		if (frameCount <= startFrame - 1) 
		{
			//cout << centroids << endl;
			std::swap(prevgray, gray);
			frameCount++;
			continue;
		}

		Mat frame1 = frame.clone();

		PutText(frame1, std::to_string(frameCount), Point(0, 100));

		//Position of mouse1
		Point2d observation1 = Point2d(centroids(0, 0), centroids(0, 1));
		//Position of mouse2
		Point2d observation2 = Point2d(centroids(1, 0), centroids(1, 1));

		if (frameCount == startFrame)
		{
			observation1 = Point2d(kF11, kF12);
			observation2 = Point2d(kF21, kF22);
		}

		Point2d prediction1 = kF1.predict();
		Point2d prediction2 = kF2.predict();

		Mat mouse1 = imread("mouse1.bmp", IMREAD_COLOR);
		Mat mouse2 = imread("mouse2.bmp", IMREAD_COLOR);

		/********* Position Tuning and Object Tracking ********/
		//Correct Error Positions before data association
		/*
		if (frameCount > 8)
		{
			detectionMissingTuning(observation1, observation2, kF1, kF2, frameCount, mouse1, mouse2);
			deviatedPositionTuning(observation1, observation2, kF1, kF2, frameCount, mouse1, mouse2);
		}
		*/
		detectionMissingTuning(observation1, observation2, kF1, kF2, frameCount, mouse1, mouse2);
		deviatedPositionTuning(observation1, observation2, kF1, kF2, frameCount, mouse1, mouse2);

		int asso = HungarianMethod(observation1, observation2, prediction1, prediction2, kF1, kF2);
		drawOnFrame(asso, frameCount, mouse1, mouse2, observation1, observation2, kF1, kF2, frame1);

		cout << "Observations1£º" << observation1 << endl;
		cout << "Observations2£º" << observation2 << endl;
		cout << "Prediction1£º" << prediction1 << endl;
		cout << "Prediction2£º" << prediction2 << endl;
		cout << endl;

		/********* Calculate Foraging Time for Each Mouse ********/
		Matrix2d forage = isForaging(foodMask1, foodMask2, mouse1, mouse2);
		//cout << "Forage matrix: " << forage << endl;
		if (asso == 1122)
		{
			kF1Food1 += forage(0, 0) * deltaT;
			kF1Food2 += forage(0, 1) * deltaT;
			kF2Food1 += forage(1, 0) * deltaT;
			kF2Food2 += forage(1, 1) * deltaT;
		}
		else if (asso == 1221)
		{
			kF1Food1 += forage(1, 0) * deltaT;
			kF1Food2 += forage(1, 1) * deltaT;
			kF2Food1 += forage(0, 0) * deltaT;
			kF2Food2 += forage(0, 1) * deltaT;
		}

		PutText(frame1, "Rat1 on food 1:" + std::to_string(kF1Food1), Point(0, 120));
		PutText(frame1, "Rat1 on food 2:" + std::to_string(kF1Food2), Point(0, 140));
		PutText(frame1, "Rat2 on food 1:" + std::to_string(kF2Food1), Point(0, 160));
		PutText(frame1, "Rat2 on food 2:" + std::to_string(kF2Food2), Point(0, 180));

		//namedWindow("Result", WINDOW_AUTOSIZE);
		//imshow("Result", frame1);
		imwrite("./output/" + std::to_string(frameCount) + ".bmp", frame1);
		resize(frame1, frame1, Size(960, 540));
		writer << frame1;
		waitKey(10);  //Ã¿Ò»Ö¡ÑÓ³Ù10ºÁÃë
		std::swap(prevgray, gray);
		frameCount++;
	}
	//f1.close();
	capture.release();
	writer.release();
	system("pause");
	//getchar();
	//getchar();
	return 0;
}