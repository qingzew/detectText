#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>

#include <math.h>
#include "GraphcutSeg.h"
#include <iostream>

typedef cv::Vec<unsigned char, 3> Vec3u;

const float CGraphcutSeg::sc = 0.012;      // color normalization constant
const float CGraphcutSeg::sg = 2;         // spatial normalization constant
const float CGraphcutSeg::D = 3;
const float CGraphcutSeg::beta = 1.0f/(2.0f*D*sc*sc);
const float CGraphcutSeg::alpha = 1.0f/(2.0f*sg*sg);
const float CGraphcutSeg::CONVERT_TO_INT_KOEF = 1000.0f;     // convert to int

///////////////////////////////////////////////////////////////////
CGraphcutSeg::CGraphcutSeg()
{
}

///////////////////////////////////////////////////////////////////
CGraphcutSeg::~CGraphcutSeg()
{
}

///////////////////////////////////////////////////////////////////
void CGraphcutSeg::SetImage(const Mat &in_Image, const Mat &in_seedsImage)
{
	seedsImage = in_seedsImage;
	inputImage = in_Image;
}

///////////////////////////////////////////////////////////////////
cv::Mat &CGraphcutSeg::getResult()
{
	return segImage;
}

///////////////////////////////////////////////////////////////////
cv::Mat &CGraphcutSeg::getSeedsImage()
{
	return seedsImage;
}

///////////////////////////////////////////////////////////////////
inline float CGraphcutSeg::ColorDif(Vec3f &color1, Vec3f &color2, float spWeights)
{
	float colorWeights = norm(color1 - color2);
	colorWeights *= colorWeights;

	float edgeWeights = exp(-beta*colorWeights - alpha*spWeights);

	return edgeWeights;
}

///////////////////////////////////////////////////////////////////
void CGraphcutSeg::Segment(float in_fU)
{
	CreateGraph(in_fU);

	segImage.release();
	segImage = Mat(inputImage.rows, inputImage.cols, CV_8U, 128);

	int flow = g -> maxflow();

	int iNode = 0;
	for (int iRow = 0; iRow < inputImage.rows; iRow ++)
	{
		for (int iCol = 0; iCol < inputImage.cols; iCol ++)
		{
			if (g->what_segment(iNode) == GraphType::SOURCE)
			{
				segImage.at <unsigned char> (iRow, iCol) = 0;
			}
			else
			{
				segImage.at <unsigned char> (iRow, iCol) = 255;
			}
			iNode ++;
		}
	}

#ifdef DEBUG_PRINTOUT
	// DEBUG PRINT_OUT
	std::string path = "seg.png";
	imwrite(path, segImage);
#endif

	delete g;
}


///////////////////////////////////////////////////////////////////
void CGraphcutSeg::CreateGraph(float in_fU)
{
	g = new GraphType(/*estimated # of nodes*/ inputImage.cols*inputImage.rows,
					  /*estimated # of edges*/ 2*inputImage.cols*inputImage.rows);

	int iNode = 0;
	for (int iRow = 0; iRow < inputImage.rows; iRow ++)
	{
		for (int iCol = 0; iCol < inputImage.cols; iCol ++)
		{
			g -> add_node();

			float curProb = seedsImage.at <float> (iRow, iCol);

			if (curProb >= 129.0/255.0)
			{
				curProb = 0.5 + (curProb - 0.5);
				g -> add_tweights( iNode, CONVERT_TO_INT_KOEF*in_fU*curProb, CONVERT_TO_INT_KOEF*in_fU*(1.0-curProb));
			}

			else if (curProb <= 127.0/255.0)
			{
				curProb = 0.5 + (0.5 - curProb);
				g -> add_tweights( iNode, CONVERT_TO_INT_KOEF*in_fU*(1.0-curProb), CONVERT_TO_INT_KOEF*in_fU*curProb );
			}

			if ((iCol == 0) || (iRow == 0))
			{
				iNode ++;
				continue;
			}

			Vec3f &color_center = inputImage.at <Vec3f> (iRow, iCol);
			Vec3f &color_top = inputImage.at <Vec3f> (iRow-1, iCol);

            int capacity = CONVERT_TO_INT_KOEF*ColorDif(color_center, color_top, 1)+1;

			g -> add_edge( iNode-inputImage.cols, iNode, capacity, capacity );

			Vec3f &color_left = inputImage.at <Vec3f> (iRow, iCol-1);
			capacity = CONVERT_TO_INT_KOEF*ColorDif(color_center, color_left, 1)+1;
			g -> add_edge( iNode-1, iNode, capacity, capacity );

			Vec3f &color_topleft = inputImage.at <Vec3f> (iRow-1, iCol-1);
			capacity = CONVERT_TO_INT_KOEF*ColorDif(color_center, color_topleft, 2)+1;
			g -> add_edge( iNode-inputImage.cols-1, iNode, capacity, capacity );

			capacity = CONVERT_TO_INT_KOEF*ColorDif(color_top, color_left, 2)+1;
			g -> add_edge( iNode-inputImage.cols, iNode-1, capacity, capacity );

			if (capacity  < 1e-6)
			{
				float tmp = 0;
			}


			iNode ++;
		}
	}
}

