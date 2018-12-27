//C++ libraries
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "utility.h"
#include "Eigen/Dense"
#include "KFilter.h"

using namespace std;
using namespace cv;
using namespace Eigen;

/*
	observation1, previousPosition1
	observation2, previousPosition2
	1111: 1-1, 2-2
	1221:
		previousPosition1: previous position of kF2
		previousPosition2: previous position of kF1
*/
bool swappingDetection(Point2d &observation1, Point2d &observation2, Point2d previousPosition1, Point2d previousPosition2)
{
	double d1 = getDistance(observation1, previousPosition1);
	double d2 = getDistance(observation2, previousPosition2);

	double d3 = getDistance(observation1, previousPosition2);
	double d4 = getDistance(observation2, previousPosition1);

	cout << "previous Position 1" << previousPosition1 << endl;
	cout << "previous Position 2" << previousPosition2 << endl;
	cout << "d1: " << d1 << endl;
	cout << "d2: " << d2 << endl;

	//The positions swapped
	if (d1 > 90 && d2 > 90)
	{
		cout << "d3: " << d3 << endl;
		cout << "d4: " << d4 << endl;
		//Point2d temp = observation1;
		//observation1 = observation2;
		//observation2 = temp;
		cout << "Swapping detected!" << endl;
		return true;
	}
	return false;
}

//Method to handle the situation that detection is missing
//Replace the observation and the mouse detected this frame is useless, have to use the mouse detection from the last frame.
//The actual result is opposite to the expected result, strange......(Missed Greedy Part)
int detectionMissingTuning(Point2d &observation1, Point2d &observation2, KFilter &kF1, KFilter &kF2, int frameCount, Mat &mouse1, Mat &mouse2)
{
	Point2d kF1position = kF1.getLastPositions().back();
	Point2d kF2position = kF2.getLastPositions().back();

	double d11 = getDistance(observation1, kF1position);
	double d12 = getDistance(observation1, kF2position);
	double d21 = getDistance(observation2, kF1position);
	double d22 = getDistance(observation2, kF2position);

	if (observation1 == Point2d(-1, -1))
	{
		//ob1不可信，ob2可信
		if (d21 <= d22) {
			cout << "Missing Objects Detected!" << endl;
			//kF1 to ob2, kF2 to ob1
			observation1 = kF2position;
			mouse1 = imread("./kF2/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
			return 1122;
		}
		else
		{
			cout << "Missing Objects Detected!" << endl;
			//kF1 to ob1, kF2 to ob2
			observation1 = kF1position;
			mouse1 = imread("./kF1/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
			return 1221;
		}
	}
	else if (observation2 == Point2d(-1, -1))
	{
		if (d11 <= d12) 
		{
			//kF1 to ob1, kF2 to ob2
			observation2 = kF2position;
			mouse2 = imread("./kF2/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
			return 1122;
		}
		else
		{
			//kF1 to ob2, kF2 to ob1
			observation2 = kF1position;
			mouse2 = imread("./kF1/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
			return 1221;
		}
	}
}

void deviatedPositionTuning(Point2d &observation1, Point2d &observation2, KFilter &kF1, KFilter &kF2, int frameCount, Mat &mouse1, Mat &mouse2)
{
	Point2d kF1position = kF1.getLastPositions().back();
	Point2d kF2position = kF2.getLastPositions().back();

	double d11 = getDistance(observation1, kF1position);
	double d12 = getDistance(observation1, kF2position);
	double d21 = getDistance(observation2, kF1position);
	double d22 = getDistance(observation2, kF2position);

	//No observation is within the gating area of kF1
	if (d11 > 80 && d21 > 80)
	{
		//Observation1 to kF2
		if (d12 <= d22)
		{
			cout << "Observation2 Wasted! KF1" << endl;
			observation2 = kF1position;
			mouse2 = imread("./kF1/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
		}
		else
		{
			cout << "Observation1 Wasted! KF1" << endl;
			observation1 = kF1position;
			mouse1 = imread("./kF1/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);
		}
	}
	//No observation is within the gating area of kF2
	else if (d12 > 80 & d22 > 80)
	{
		//Observation1 to kF1
		if (d11 <= d21)
		{
			cout << "Observation2 Wasted! KF2" << endl;
			observation2 = kF2position;
			mouse2 = imread("./kF2/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);;
		}
		else
		{
			cout << "Observation1 Wasted! KF2" << endl;
			observation1 = kF2position;
			mouse1 = imread("./kF2/" + std::to_string(frameCount - 1) + ".bmp", IMREAD_COLOR);;
		}
	}
}

//Data Association Method. Since we know there are no more than 2 objects, we simplified the implementation here.
int HungarianMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, KFilter kF1, KFilter kF2)
{
	//PreviousPosition
	Point2d p1 = kF1.getLastPositions().back();
	Point2d p2 = kF2.getLastPositions().back();

	//One possible combination
	double d11 = getDistance(observation1, prediction1);
	double d22 = getDistance(observation2, prediction2);
	//Another possible combination
	double d12 = getDistance(observation1, prediction2);
	double d21 = getDistance(observation2, prediction1);

	double sum1 = d11 + d22;
	double sum2 = d12 + d21;

	//Hungarian Method, shortest overall distance
	if (sum1 <= sum2) //1122, kF1 to mouse1, kF2 to mouse2
	{   
		//Data Tuning
		if (swappingDetection(observation1, observation2, p1, p2))
		{
			cout << "Tuning: 1122 to 1221" << endl;
			//True, corrected, should be 1221
			return 1221;
		}
		else
			return 1122;
	}
	else //1221, kF1 to mouse2, kF2 to mouse1
	{
		//Data Tuning
		if (swappingDetection(observation1, observation2, p2, p1))
		{
			//True, corrected, should be 1122
			cout << "Tuning: 1221 to 1122" << endl;
			return 1122;
		}
		else
			return 1221;
	}

	cout << "d11: " << d11 << endl;
	cout << "d22: " << d22 << endl;
	cout << "d12: " << d12 << endl;
	cout << "d21: " << d21 << endl;

	cout << " ################################" << endl;
}

int GreedyMethod(Point2d observation1, Point2d observation2, Point2d prediction1, Point2d prediction2, KFilter kF1, KFilter kF2) {
	//One possible combination
	double d11 = getDistance(observation1, prediction1);
	double d22 = getDistance(observation2, prediction2);
	//Another possible combination
	double d12 = getDistance(observation1, prediction2);
	double d21 = getDistance(observation2, prediction1);

	//observation1 to kF1
	if (d11 <= d12)
		return 1122;
	else
		return 1221;
}

void drawOnFrame(int asso, int frameCount, Mat mouse1, Mat mouse2, Point2d observation1, Point2d observation2, KFilter &kF1, KFilter &kF2, Mat &src)
{
	if (asso == 1221)
	{
		imwrite("./kF1/" + std::to_string(frameCount) + ".bmp", mouse2);
		imwrite("./kF2/" + std::to_string(frameCount) + ".bmp", mouse1);
		drawMice(mouse1, src, kF2.getBoxColor());
		drawMice(mouse2, src, kF1.getBoxColor());
		kF1.update(pointToVector(observation2));
		kF2.update(pointToVector(observation1));
	}
	else if (asso == 1122)
	{
		imwrite("./kF1/" + std::to_string(frameCount) + ".bmp", mouse1);
		imwrite("./kF2/" + std::to_string(frameCount) + ".bmp", mouse2);
		drawMice(mouse1, src, kF1.getBoxColor());
		drawMice(mouse2, src, kF2.getBoxColor());
		kF1.update(pointToVector(observation1));
		kF2.update(pointToVector(observation2));
	}
}