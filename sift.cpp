/*
 * http://blog.csdn.net/zddblog/article/details/7521424
 */
#include <iostream>
#include <set>
#include <functional>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <fstream>
#define DEBUG

class Data{
public:
	Data(int x, int y, double d) {
		this->x = x;
		this->y = y;
		this->dis = d;
	}

	int x;
	int y;
	double dis;
};

class Comp {
public:
	bool operator()(const Data &d1, const Data &d2) {
		return d1.dis > d2.dis;
	}
};

int main(int argc, char **argv) {
	//cv::Mat img = cv::imread("a.png");
	//cv::cvtColor(img, img, CV_BGR2GRAY);

	////cv::SIFT sift(2000, 50, 0.04, 5);
	//cv::SIFT sift;
	//std::vector<cv::KeyPoint> keypoints;
	//cv::Mat desc, dest;

	////sift(img, img, keypoints, desc, false);
	//sift(img, cv::noArray(), keypoints, desc, false);

	//std::cout <<"keypoints: "  << keypoints.size() << std::endl;
	//std::cout << "desc: " << desc.size() << std::endl;
	//cv::drawKeypoints(img, keypoints, dest, cv::Scalar(0, 0, 255));

	//cv::imshow("sift", dest);

	//cv::waitKey(0);

	cv::Mat source = cv::imread("siftData/sku-1.png");
	cv::cvtColor(source, source, CV_BGR2GRAY);

	if (source.empty()) {
		std::cerr << "no image..." << std::endl;
		exit(1);
	}

	cv::Mat target = cv::imread("siftData/e.png");
	cv::cvtColor(target, target, CV_BGR2GRAY);
	if (target.empty()) {
		std::cerr << "no image..." << std::endl;
		exit(1);
	}

	cv::vector<cv::KeyPoint> keyPointS, keyPointT;
	cv::Mat descS, descT;

	//cv::SIFT  sift;
	cv::SIFT sift(2000, 15, 0.04, 0.3);

	sift(source, cv::noArray(), keyPointS, descS, false);
	sift(target, cv::noArray(), keyPointT, descT, false);

#ifdef DEBUG
	std::cout <<"keyPointS: "  << keyPointS.size() << std::endl;
	std::cout << "descS: " << descS.size() << std::endl;

	std::cout << "keyPointT: " << keyPointT.size() << std::endl;
	std::cout << "descT: " << descT.size() << std::endl;

	double minS, maxS, minT, maxT;
	cv::minMaxLoc(descS, &minS, &maxS);
	cv::minMaxLoc(descT, &minT, &maxT);

	std::cout << "minS: " << minS << std::endl;
	std::cout << "maxS: " << maxS << std::endl;
	
	std::cout << "minT: " << minT << std::endl;
	std::cout << "maxT: " << maxT << std::endl;
#endif

	////ÄÚ»ý 
	//std::set<double, std::greater<double> > s;
	//for (int i = 0; i < descS.rows; i++) {
	//	for (int j = 0; j < descT.rows; j++) {
	//		s.insert(descS.row(i).dot(descT.row(j)));
	//	}
	//}

	//Å·Ê½¾àÀë
	std::set<Data, Comp> s;
	for (int i = 0; i < descS.rows; i++) {
		for (int j = 0; j < descT.rows; j++) {
			cv::Mat p;
			cv::pow(descS.row(i) - descT.row(j), 2.0, p);
			double d = sqrt(cv::sum(p)[0]);
			s.insert(Data(i, j, d));

		}
	}

#ifdef DEBUG
	std::vector<cv::KeyPoint> ks, kt;

	int cnt = 0;
	for (std::set<Data, Comp>::reverse_iterator it = s.rbegin(); it != s.rend(); it++) {
		std::cout << "dis: " << (*it).dis << std::endl;
		ks.push_back(keyPointS[(*it).x]);
		kt.push_back(keyPointT[(*it).y]);

		if (++cnt >= 20) {
			break;
		}
	}

	
	cv::Mat ds, dt;
	cv::drawKeypoints(source, ks, ds, cv::Scalar(0, 0, 255));
	cv::drawKeypoints(target, kt, dt, cv::Scalar(0, 0, 255));

	cv::imshow("source", ds);
	cv::imshow("target", dt);
	cv::waitKey(0);

#endif

	return 0;
}