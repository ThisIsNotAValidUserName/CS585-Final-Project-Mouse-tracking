//C++ libraries
#include <iostream> 
#include <cmath>

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//Custom Libraries
#include "Eigen/Dense"
#include "KFilter.h"

using namespace Eigen;
using namespace std;
using namespace cv;

KFilter::KFilter(Scalar boxColor) {
	this->boxColor = boxColor;
	initialize();
}

KFilter::~KFilter() {

}

void KFilter::initialize() {
	A << 1, 0, deltaT, 0,
		0, 1, 0, deltaT,
		0, 0, 1, 0,
		0, 0, 0, 1;

	B << (double) (pow(deltaT, 2) / 2), (double) (pow(deltaT, 2) / 2), deltaT, deltaT;

	H = MatrixXd(measureNum, stateNum);
	H << 1, 0, 0, 0,
		0, 1, 0, 0;

	R << 1.0, 0,
		0, 1.0;

	//Q = B * BT
	Q << pow(deltaT, 4) / 4, pow(deltaT, 4) / 4, pow(deltaT, 3) / 2, pow(deltaT, 3) / 2,
		pow(deltaT, 4) / 4, pow(deltaT, 4) / 4, pow(deltaT, 3) / 2, pow(deltaT, 3) / 2,
		pow(deltaT, 3) / 2, pow(deltaT, 3) / 2, pow(deltaT, 2), pow(deltaT, 2),
		pow(deltaT, 3) / 2, pow(deltaT, 3) / 2, pow(deltaT, 2), pow(deltaT, 2);

	Q = Q * 0.001;
		
	/*
	Q << 0.1, 0, 0, 0,
		0, 0.1, 0, 0,
		0, 0, 0.1, 0,
		0, 0, 0, 0.1;
		*/

	P_init << 10, 0, 0, 0,
		0, 10, 0, 0,
		0, 0, 10, 0,
		0, 0, 0, 10;

	Pk_1 = P_init;
	//X_1 all zero for start, could change later
	xk_1 << 0, 0, 0, 0;
	lastPositions.push_back(Point2d(0, 0));
}

Point2d KFilter::predict() 
{
	//predict (Time update)
	x_k = A * xk_1 + B * u; //(A * xk_1 + Buk)
	//x_k = A * xk_1;
	P_k = A * Pk_1 * A + Q;  //(A * Pk_1 * AT + Q)
	return Point2d(x_k(0), x_k(1));
}

void KFilter::update(Vector2d z) 
{
	//update (Measurement Update)
	MatrixXd S_k = H * P_k * H.transpose() + R;
	K_k = P_k * H.transpose() * S_k.inverse();  //(P_k * HT) * [(H * P_k * HT + R)]-1
	x_k = x_k + K_k * (z - H * x_k);
	P_k = (identityMatrix - K_k * H) * P_k;   //(I - K_k * H) * P_k

	//x_k is what we want
	xk_1 = x_k;
	Pk_1 = P_k;

	//lastPosition
	lastPositions.push_back(Point2d(z(0), z(1)));
}

void KFilter::setInitialState(double x, double y) 
{
	xk_1 << x, y, 0, 0;
	lastPositions.push_back(Point2d(xk_1(0), xk_1(1)));
}

vector<Point2d> KFilter::getLastPositions() 
{
	return lastPositions;
}

Scalar KFilter::getBoxColor() 
{
	return boxColor;
}