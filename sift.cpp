#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, char **argv) {
	cv::Mat img = cv::imread("41.png");
	cv::cvtColor(img, img, CV_BGR2GRAY);

	//cv::SIFT sift(2000, 50, 0.04, 5);
	cv::SIFT sift;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat desc, dest;

	//sift(img, img, keypoints, desc, false);
	sift(img, cv::noArray(), keypoints, desc, false);

	std::cout <<"keypoints: "  << keypoints.size() << std::endl;

	cv::drawKeypoints(img, keypoints, dest, cv::Scalar(0, 0, 255));

	cv::imshow("sift", dest);

	cv::waitKey(0);

	return 0;
}