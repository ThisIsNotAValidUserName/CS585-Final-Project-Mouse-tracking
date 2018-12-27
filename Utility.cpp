//C++ libraries
#include <vector>
#include <iostream>

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

Matrix2d getMice(Mat flow, Mat src, Mat flowthres) {
	//Result
	Matrix2d centroid_s;
	int erosion_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(erosion_size, erosion_size));
	dilate(flowthres, flowthres, element);
	bitwise_not(flowthres, flowthres);
	Mat src_gray, thres_output, result1;
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	Rect area = Rect(0.187*src.cols, 0.2*src.rows, 0.68*src.cols, 0.739*src.rows);
	mask(area).setTo(255);
	bitwise_not(mask, mask);
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	src_gray.setTo(255, flowthres);
	src_gray.setTo(255, mask);
	threshold(src_gray, thres_output, 70, 255, 0);

	bitwise_not(thres_output, thres_output);
	cv::erode(thres_output, thres_output, element);
	//cv::erode(thres_output, thres_output, element);
	cv::dilate(thres_output, thres_output, element);

	//imshow("Thres", thres_output);

	bool flag = findMiceByColor(src, src_gray, thres_output, centroid_s);
	if (!flag) {
		threshold(src_gray, thres_output, 30, 255, 0);
		centroid_s = devideFromFlow(flow, thres_output, src);
	}
	return centroid_s;
}

bool findMiceByColor(Mat &src, Mat &src_gray, Mat &thres_output, Matrix2d &centroid_s) {
	Mat mice1, mice2;
	cv::Mat  labels, stats, centroids;
	//Mat image = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int nccomps = cv::connectedComponentsWithStats(
		thres_output, labels,
		stats, centroids
	);

	if (nccomps > 2) {
		//find largest 2 components and draw rectangles
		int max1 = 1;
		int max2 = 2;

		if (stats.at<int>(max2, cv::CC_STAT_AREA) > stats.at<int>(max1, cv::CC_STAT_AREA)) {
			max1 = 2;
			max2 = 1;
		}
		for (int i = 3; i < nccomps; i++) {
			if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(max1, cv::CC_STAT_AREA)) {
				max2 = max1;
				max1 = i;
			}
			else if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(max2, cv::CC_STAT_AREA)) {
				max2 = i;
			}
		}
		cout << "area" << stats.at<int>(max1, cv::CC_STAT_AREA) << endl;
		if (stats.at<int>(max1, cv::CC_STAT_AREA) > 10000) {
			Mat mask = Mat::zeros(src.size(), CV_8UC1);
			for (int i = 0; i < labels.cols; i++) {
				for (int j = 0; j < labels.rows; j++) {
					int label = labels.at<int>(j, i);
					if (label == max1) {
						mask.at<uchar>(j, i) = 255;
					}
				}
			}
			//src_gray = src_gray(mask);
			src_gray.copyTo(src_gray, mask);
			return false;
		}
		src.copyTo(mice1);
		src.copyTo(mice2);

		for (int i = 0; i < labels.cols; i++) {
			for (int j = 0; j < labels.rows; j++) {
				int label = labels.at<int>(j, i);
				if (label == max1) {
					mice1.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
				}
				else
				{
					mice1.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
				}

				if (label == max2) {
					mice2.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
				}
				else
				{
					mice2.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
				}

			}
		}

		imwrite("mouse1.bmp", mice1);
		imwrite("mouse2.bmp", mice2);

		centroid_s << centroids.at<double>(max1, 0), centroids.at<double>(max1, 1),
			centroids.at<double>(max2, 0), centroids.at<double>(max2, 1);
		return true;
	}
	return false;
}

void drawMice(Mat &image, Mat &src, Scalar boxColor) 
{
	cv::Mat labels, stats, centroids;
	cvtColor(image, image, CV_BGR2GRAY);
	threshold(image, image, 0, 255, 0);
	bitwise_not(image, image);
	int nccomps = cv::connectedComponentsWithStats(
		image, labels,
		stats, centroids
	);
	//: Find largest component
	int maxsize = 0;
	int maxind = 0;
	for (int i = 1; i < nccomps; i++)
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
		}
	}
	double left, top, height, width;
	left = stats.at<int>(maxind, CC_STAT_LEFT);
	top = stats.at<int>(maxind, CC_STAT_TOP);
	width = stats.at<int>(maxind, CC_STAT_WIDTH);
	height = stats.at<int>(maxind, CC_STAT_HEIGHT);
	Rect rect1(left, top, width, height);
	rectangle(src, rect1, boxColor, 5, 8, 0);
}

Matrix2d isForaging(Mat foodMask1, Mat foodMask2, Mat mouseMask1, Mat mouseMask2)
{
	Mat dst1, dst2, dst3, dst4;

	bitwise_not(mouseMask1, mouseMask1);
	bitwise_not(mouseMask2, mouseMask2);

	bitwise_and(mouseMask1, foodMask1, dst1);
	bitwise_and(mouseMask1, foodMask2, dst2);
	bitwise_and(mouseMask2, foodMask1, dst3);
	bitwise_and(mouseMask2, foodMask2, dst4);

	//cout << "##################################" << endl;
	//cout << "Combination t1: " << endl;
	int t1 = imageAllBlack(dst1);
	//cout << "Combination t2: " << endl;
	int t2 = imageAllBlack(dst2);
	//cout << "Combination t3: " << endl;
	int t3 = imageAllBlack(dst3);
	//cout << "Combination t4: " << endl;
	int t4 = imageAllBlack(dst4);
	//cout << "##################################" << endl;

	Matrix2d result;
	result << t1, t2,
		t3, t4;
	return result;
}

/***** Distance between two points *****/
double getDistance(Point2d pointO, Point2d pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}

Vector2d pointToVector(Point2d point) {
	Vector2d vector;
	vector << point.x, point.y;
	return vector;
}

void PutText(Mat &src, string text, Point origin) {
	//Point origin(0, 100);
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 0.5;
	int thickness = 1;
	putText(src, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
}

Matrix2d devideFromFlow(Mat flow, Mat &thres_output, Mat &src) {
	Matrix2d centroid_s;
	Mat mask1 = Mat::zeros(flow.size(), CV_8UC1);
	Mat mask2 = Mat::zeros(flow.size(), CV_8UC1);
	bitwise_not(thres_output, thres_output);
	vector<float> ori;
	vector<Point> points;
	for (int y = 0; y < flow.rows; y += 1)
		for (int x = 0; x < flow.cols; x += 1)
		{
			if (thres_output.at<uchar>(y, x) == 255) {
				const Point2f& fxy = flow.at<Point2f>(y, x);
				float param = fxy.y / fxy.x;
				float result = atan(param) * 180 / 3.14159265;
				//cout << result << endl;
				ori.push_back(result);
				points.push_back(Point(x, y));
			}
		}
	int i1 = ori[0];
	int i2 = ori[10];
	//分群1，2
	vector<float>g1, g2;
	vector<int>index1, index2;
	//c1,c2中心
	int c1 = 0, c2 = 0;
	while (i1 != c1 || i2 != c2)
	{
		c1 = i1;
		c2 = i2;

		g1.clear();
		g2.clear();
		index1.clear();
		index2.clear();
		for (int i = 0; i < ori.size(); i++)
		{
			float abs1 = abs(i1 - ori[i]);
			float abs2 = abs(i2 - ori[i]);
			if (abs1 <= abs2) {
				g1.push_back(ori[i]);
				index1.push_back(i);
			}
			else {
				index2.push_back(i);
				g2.push_back(ori[i]);
			}

		}
		float sum = 0;
		for (int i = 0; i < g1.size(); i++) {
			sum += g1[i];
		}
		i1 = sum / g1.size();
		sum = 0;
		for (int i = 0; i < g2.size(); i++) {
			sum += g2[i];
		}
		i2 = sum / g2.size();
	}

	for (int i = 0; i < index1.size(); i++) {
		Point p = points[index1[i]];
		mask1.at<uchar>(p) = 255;
	}
	for (int i = 0; i < index2.size(); ++i) {
		Point p = points[index2[i]];
		mask2.at<uchar>(p) = 255;
	}
	Point2d p1 = findCentroid(mask1);
	Point2d p2 = findCentroid(mask2);
	centroid_s << p1.x, p1.y,
		p2.x, p2.y;
	bitwise_not(mask1, mask1);
	bitwise_not(mask2, mask2);
	imwrite("mouse1.bmp", mask1);
	imwrite("mouse2.bmp", mask2);
	return centroid_s;
}

Point2d findCentroid(Mat &image) {
	cv::Mat  labels, stats, centroids;
	//resize(image, srcout, Size(500, 300));
	//imshow("iii", srcout);
	int nccomps = cv::connectedComponentsWithStats(
		image, labels,
		stats, centroids
	);
	//: Find largest contour
	int maxsize = 0;
	int maxind = 0;
	for (int i = 1; i < nccomps; i++)
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
		}
	}
	if (nccomps > 1) {
		return Point2d(centroids.at<double>(maxind, 0), centroids.at<double>(maxind, 1));
	}
	else {
		return Point2d(-1, -1);
	}
}

int imageAllBlack(Mat src)
{
	int area = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
			{
				area += 1;
			}
		}
	}

	cout << "Area: " << area << endl;

	if (area > 100)
		return 1;
	else
		return 0;
}