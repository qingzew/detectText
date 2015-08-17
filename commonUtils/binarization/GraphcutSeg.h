#pragma once

#include <opencv2/core/core.hpp>
#include "../maxflow/graph.h"


using namespace cv;


typedef Graph<int,int,int> GraphType;


class CGraphcutSeg
{
public:

	CGraphcutSeg();
	virtual ~CGraphcutSeg();

	void SetImage(const Mat &in_Image, const Mat &in_seedsImage);
	void Segment(float in_U);
	cv::Mat &getResult();
	cv::Mat &getSeedsImage();

private:

	inline float ColorDif(Vec3f &color1, Vec3f &color2, float spWeights);
	void CreateGraph(float in_fU);

	Mat segImage;
	Mat	seedsImage;
	Mat inputImage;

	GraphType *g;
	float U;       // unary coefficient

	static const float sc;      // color normalization constant
	static const float sg;         // spatial normalization constant
	static const float D;
	static const float beta;
	static const float alpha;
	static const float CONVERT_TO_INT_KOEF; // term for converting to integer

};
