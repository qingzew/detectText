#include <iostream>
#include "binarization.h"
#include "commonUtils/binarization/GraphcutSeg.h"

#include <fstream>

const double k =  0.5;
const double dR = 128;

void PM_G1(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k)
{
    //cv::exp(-(Lx.mul(Lx) + Ly.mul(Ly))/(k*k),dst);
    int N = Lx.rows * Lx.cols;
    float lx = 0.0, ly = 0.0, k2 = k*k;


    for (int i=0; i<N; i++)
    {
        lx = *(Lx.ptr<float>(0)+i);
        ly = *(Ly.ptr<float>(0)+i);
        lx *= lx;
        ly *= ly;
        *(dst.ptr<float>(0)+i) = std::exp(-(lx + ly)/k2);
    }


}

void PM_G2(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k )
{
    //dst = 1./(1. + (Lx.mul(Lx) + Ly.mul(Ly))/(k*k));
    int N = Lx.rows * Lx.cols;
    float lx = 0.0, ly = 0.0, k2 = k*k;

    for (int i = 0; i < N; i++)
    {
        lx = *(Lx.ptr<float>(0)+i);
        ly = *(Ly.ptr<float>(0)+i);
        lx *= lx;
        ly *= ly;
        *(dst.ptr<float>(0)+i) = 1.0 / (1.0 + (lx + ly)/k2);
    }

}

void Weickert_Diffusivity(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k)
{
    //cv::Mat modg;
    //cv::pow((Lx.mul(Lx) + Ly.mul(Ly))/(k*k),4,modg);
    //cv::exp(-3.315/modg, dst);
    //dst = 1.0 - dst;

    int N = Lx.rows * Lx.cols;
    float lx2 = 0.0, ly2 = 0.0, modg = 0.0;
    const float k2 = k*k;

    for (int i = 0; i < N; i++)
    {
        lx2 = *(Lx.ptr<float>(0)+i);
        ly2 = *(Ly.ptr<float>(0)+i);
        lx2 *= lx2;
        ly2 *= ly2;
        modg = std::pow( (lx2 + ly2)/k2, 4 );
        *(dst.ptr<float>(0)+i) = 1.0 - std::exp( -3.315/modg );
    }

}

void Gaussian_2D_Convolution(const cv::Mat &src, cv::Mat &dst, unsigned int ksize_x,
                             unsigned int ksize_y, float sigma)
{
    // Compute an appropriate kernel size according to the specified sigma
    if( sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0 )
    {
        ksize_x = ceil(2.0*(1.0 + (sigma-0.8)/(0.3)));
        ksize_y = ksize_x;
    }

    // The kernel size must be and odd number
    if( (ksize_x % 2) == 0 )
    {
        ksize_x += 1;
    }

    if( (ksize_y % 2) == 0 )
    {
        ksize_y += 1;
    }

    // Perform the Gaussian Smoothing with border replication
    cv::GaussianBlur(src,dst,cv::Size(ksize_x,ksize_y),sigma,sigma,cv::BORDER_REPLICATE);

}

void Image_Derivatives_Scharr(const cv::Mat &src, cv::Mat &dst, unsigned int xorder, unsigned int yorder)
{
   // Compute Scharr filter
   cv::Scharr(src,dst,CV_32F,xorder,yorder,1,0,cv::BORDER_DEFAULT);
}


float Compute_K_Percentile(const cv::Mat &img, float perc, float gscale, unsigned int nbins, unsigned int ksize_x, unsigned int ksize_y)
{
    float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
    unsigned int nbin = 0, nelements = 0, nthreshold = 0, k = 0;
    float hmax = 0.0;        // maximum gradient
    int npoints = 0.0;    // number of points of which gradient greater than zero

    // Create the array for the histogram
    std::vector<float> hist(nbins,0);
    std::vector<float> Mo;

    // Create the matrices
    cv::Mat gaussian = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    cv::Mat Lx = cv::Mat::zeros(img.rows,img.cols,CV_32F);
    cv::Mat Ly = cv::Mat::zeros(img.rows,img.cols,CV_32F);

    // Perform the Gaussian convolution
    Gaussian_2D_Convolution(img,gaussian,ksize_x,ksize_y,gscale);

    // Compute the Gaussian derivatives Lx and Ly
    Image_Derivatives_Scharr(gaussian,Lx,1,0);
    Image_Derivatives_Scharr(gaussian,Ly,0,1);

    // Get the maximum
    cv::Mat Lx1 = Lx.rowRange(1,Lx.rows-1).colRange(1,Lx.cols-1);
    cv::Mat Ly1 = Ly.rowRange(1,Ly.rows-1).colRange(1,Ly.cols-1);
    int N = Lx1.rows*Lx1.cols;

    for( int j = 0; j < N; j++ )
    {
        lx = *(Lx.ptr<float>(0)+j);
        ly = *(Ly.ptr<float>(0)+j);
        if (!lx && !ly)
            continue;

        modg = sqrt(lx*lx + ly*ly);

        Mo.push_back(modg);
    }

    hmax = *std::max_element(Mo.begin(), Mo.end());

    // Compute the histogram
    float hmax1 = 1.00001*hmax;
    npoints = Mo.size();

    for (int i = 0; i < npoints; i++)
    {
        nbin = floor(nbins*(Mo[i]/hmax1));

        hist[nbin]++;
    }

    // Now find the perc of the histogram percentile
    nthreshold = (unsigned int)(npoints*perc);

    // find the bin (k) in which accumulated points are greater than 70% (perc) of total valid points (npoints)
    for( k = 0; nelements < nthreshold && k < nbins; k++)
    {
        nelements = nelements + hist[k];
    }

    if( nelements < nthreshold )
    {
        kperc = 0.03;
    }
    else
    {
        kperc = hmax*(k/(float)nbins);
    }

    return kperc;
}

cv::Mat mat2gray(const Mat& src)
{
	Mat dst;
	normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
	return dst;
}

cv::Mat gradient(Mat& image) {
    float s[15] = { -1, -2, 0, 2, 1,
					-2, -4, 0, 4, 2,
					-1, -2, 0, 2, 1 };

	cv::Mat kernel = 1.0 / 32.0 *  cv::Mat(3, 5, CV_32F, s);
    cv::Mat kernelT = kernel.t();


//    cv::Mat kernelF;
//    cv::flip(kernel, kernelF, -1);
//    cv::Mat kernelTF;
//    cv::flip(kernel, kernelTF, -1);
//
//    cv::Point anchor(kernelT.cols-kernelT.cols/2-1, kernelT.rows-kernelT.rows/2-1);
//    cv::Point anchorT(kernelTF.cols-kernelTF.cols/2-1, kernelTF.rows-kernelTF.rows/2-1);
//
//    cv::Mat gradx, grady, grad;
//    cv::filter2D(image, gradx, image.depth(), kernelF, cv::Point(-1, -1), 0, BORDER_CONSTANT);
//    cv::filter2D(image, grady, image.depth(), kernelTF, cv::Point(-1, -1), 0, BORDER_CONSTANT);
//
//    cv::addWeighted(gradx, 0.5, grady, 0.5, 0, grad);

//    cv::Mat gradx, grady, grad;
//    cv::filter2D(image, gradx, CV_32F, kernel);
//    cv::filter2D(image, grady, CV_32F, kernelT);


//    cv::addWeighted(gradx, 0.5, grady, 0.5, 0, grad);
//    cv::imshow("grad", grad);

    cv::Mat gradx, grady;
    cv::Mat grad(image.rows, image.cols, CV_32F);
    cv::filter2D(image, gradx, CV_32F, kernel);
    cv::filter2D(image, grady, CV_32F, kernelT);

    double k = Compute_K_Percentile(image, 0.7, 1, 300, 0, 0);
    std::cout<<k<<std::endl;
//    PM_G2(image, grad, gradx, grady, k);
    Weickert_Diffusivity(image, grad, gradx, grady, k);
    cv::imshow("grad", grad);


    return grad;
}

//return mean and deviation
void calMeanDev(cv::Mat image, cv::Mat &mean, cv::Mat &deviation) {
	if (image.empty()) {
		std::cerr << "image is empty" << std::endl;
		exit(1);
	}

	int winy = (int)(2.0 * image.rows - 1) / 3;
	int winx = (int)image.cols - 1 < winy ? image.cols - 1 : winy;
	if (winx > 100)	{
		winx = winy = 40;
	}

	cv::Mat image32f;
	image.convertTo(image32f, CV_32F);

	cv::blur(image32f, mean, Size(winx, winy));

	cv::Mat meanSqu;
	cv::blur(image32f.mul(image32f), meanSqu, Size(winx, winy));

	cv::sqrt(meanSqu - mean.mul(mean), deviation);

	//imshow("coke", mat2gray(image32f));
	//imshow("mu", mat2gray(mu));
	//imshow("sigma", mat2gray(sigma));
}

cv::Mat binNiblack(cv::Mat image, double k) {
	if (image.empty()) {
		std::cout << "image is empty" << std::endl;
		exit(1);
	}
	cv::Mat mean, deviation;
	calMeanDev(image, mean, deviation);

	if (mean.empty() || deviation.empty()) {
		std::cout << "mean or deviation is empty" << std::endl;
		exit(1);
	}

	Mat thr = mean + k * deviation;
	Mat thr_ = mean - k * deviation;

	Mat output = Mat::ones(image.rows, image.cols, CV_32F) * 0.5;
	for (int j = 0; j < image.cols; j++) {
		for (int i = 0; i < image.rows; i++) {
			if (image.at<uchar>(i, j) > thr.at<float>(i, j)) {
				output.at<float>(i, j) = 1;
			}
			if (image.at<uchar>(i, j) < thr_.at<float>(i, j)) {
				output.at<float>(i, j) = 0;
			}
		}
	}

    return output;
}

cv::Mat binSauvola(cv::Mat image, double k, double dR) {
	if (image.empty()) {
		std::cout << "image is empty" << std::endl;
		exit(1);
	}

	cv::Mat mean, deviation;
	calMeanDev(image, mean, deviation);

	cv::Mat image_ = 255 - image;
	cv::Mat mean_, deviation_;
	calMeanDev(image_, mean_, deviation_);

	if (mean.empty() || deviation.empty() || mean_.empty() || deviation_.empty()) {
		std::cout << "mean or deviation is empty" << std::endl;
		exit(1);
	}

	cv::Mat thr = mean.mul(1 + 0.5 * (deviation / dR - 1));
	cv::Mat thr_ = mean_.mul(1 + 0.5 * (deviation_ / dR - 1));

	Mat output = Mat::ones(image.rows, image.cols, CV_32F) * 0.5;
	for (int j = 0; j < image.cols; j++) {
		for (int i = 0; i < image.rows; i++) {

			if (image.at<uchar>(i, j) > thr.at<float>(i, j) && image_.at<uchar>(i, j) <= thr_.at<float>(i, j)) {
				output.at<float>(i, j) = 1;
			}
			if (image_.at<uchar>(i, j) > thr_.at<float>(i, j) && image.at<uchar>(i, j) <= thr.at<float>(i, j)) {
				output.at<float>(i, j) = 0;
			}
		}
	}

    return output;
}

cv::Mat binWolf(cv::Mat image, double k, double dR) {
	if (image.empty()) {
		std::cout << "image is empty" << std::endl;
		exit(1);
	}

	cv::Mat mean, deviation;
	calMeanDev(image, mean, deviation);

	cv::Mat image_ = 255 - image;
	cv::Mat mean_, deviation_;
	calMeanDev(image_, mean_, deviation_);

	if (mean.empty() || deviation.empty() || mean_.empty() || deviation_.empty()) {
		std::cout << "mean or deviation is empty" << std::endl;
		exit(1);
	}

	double minG, maxG, minG_, maxG_, minD, maxD, minD_, maxD_;
	cv::minMaxLoc(image, &minG, &maxG);
	cv::minMaxLoc(image_, &minG_, &maxG_);
	cv::minMaxLoc(image, &minD, &maxD);
	cv::minMaxLoc(image_, &minD_, &maxD_);

	cv::Mat thr = mean + k * (deviation / maxD - 1).mul(mean - minG);
	cv::Mat thr_ = mean_ + k * (deviation_ / maxD_ - 1).mul(mean_ - minG);

	Mat output = Mat::ones(image.rows, image.cols, CV_32F) * 0.5;
	for (int j = 0; j < image.cols; j++) {
		for (int i = 0; i < image.rows; i++) {
			if (image.at<uchar>(i, j) < thr.at<float>(i, j) && image_.at<uchar>(i, j) >= thr_.at<float>(i, j)) {
				output.at<float>(i, j) = 0;
			}
			if (image_.at<uchar>(i, j) < thr_.at<float>(i, j) && image.at<uchar>(i, j) >= thr.at<float>(i, j)) {
				output.at<float>(i, j) = 1;
			}
		}
	}

    return output;
}



cv::Mat gcnlBin(cv::Mat image, std::string method) {
	if (image.empty()) {
		std::cout << "image is empty" << std::endl;
		exit(1);
	}

	cv::Mat gray(image.rows, image.cols, CV_32F);
	cv::cvtColor(image, gray, CV_BGR2GRAY);

	//cv::GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

    cv::Mat lap = gradient(gray);
    lap.convertTo(lap, CV_32F);
	lap = cv::abs(lap);

	double max = 0, min = 0;
	cv::minMaxLoc(lap, &min, &max);

	cv::Mat seedImg = (1.0 + lap / max) / 2.0;

	seedImg.row(0) = 0.5;
	seedImg.row(1) = 0.5;
	seedImg.row(2) = 0.5;
	seedImg.row(seedImg.rows - 3) = 0.5;
	seedImg.row(seedImg.rows - 2) = 0.5;
	seedImg.row(seedImg.rows - 1) = 0.5;

	seedImg.col(0) = 0.5;
	seedImg.col(1) = 0.5;
	seedImg.col(2) = 0.5;
	seedImg.col(seedImg.cols - 3) = 0.5;
	seedImg.col(seedImg.cols - 2) = 0.5;
	seedImg.col(seedImg.cols - 1) = 0.5;

	cv::Mat absSeedImg = cv::abs(seedImg - 0.5);

	cv::Mat binSeed;
	if (method == "n") {
		binSeed = binNiblack(gray, k);
	}
	else if (method == "s") {
		binSeed = binSauvola(gray, k, dR);
	}
	else if (method == "w") {
		binSeed = binWolf(gray, k, dR);
	}
	else {
		std::cout << "error: there is no method" << std::endl;
		exit(1);
	}

	cv::Mat binSeed1 = (binSeed < 0.5) * 1.0 / 255;
	binSeed1.convertTo(binSeed1, CV_32F);
	binSeed1 -= 0.5;

	cv::Mat binSeed2 = (binSeed > 0.5) * 1.0 / 255;
	binSeed2.convertTo(binSeed2, CV_32F);
	binSeed2 -= 0.5;

	cv::Mat inProbs1 = 0.5 + 4 * absSeedImg.mul(binSeed1);
	cv::imshow("inProbs1", inProbs1);
	cv::Mat inProbs2 = 0.5 + 4 * absSeedImg.mul(binSeed2);
	cv::imshow("inProbs2", inProbs2);

	cv::Mat image32f;
	image.convertTo(image32f, CV_32F);

	CGraphcutSeg gcs1;
	gcs1.SetImage(image32f, inProbs1);
	gcs1.Segment(0.25);
	cv::Mat labels1 = gcs1.getResult();
	cv::imshow("label1", labels1);

	CGraphcutSeg gcs2;
	gcs2.SetImage(image32f, inProbs2);
	gcs2.Segment(0.25);
	cv::Mat labels2 = gcs2.getResult();
	cv::imshow("label2", labels2);

	cv::Mat output = cv::Mat::ones(image.rows, image.cols, CV_32F);

	for (int i = 0; i<output.rows; i++) {
		for (int j = 0; j<output.cols; j++) {
			if (labels1.at<uchar>(i, j) == 255 && labels2.at<uchar>(i, j) == 0) {
				output.at<float>(i, j) = 1.0;
			}

			if (labels1.at<uchar>(i, j) == 0 && labels2.at<uchar>(i, j) == 255) {
				output.at<float>(i, j) = 0;
			}
		}
	}

	return output;
}


int main(int argc, char **argv) {
    std::string file = argv[1];
	cv::Mat image = cv::imread(file);

	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);

    //cv::Mat grad = gradient(gray);
    //cv::imshow("grad", grad);
	cv::Mat niblack = binNiblack(gray, k);
	cv::imshow("niblack", mat2gray(niblack));

//	cv::Mat sauvola = binSauvola(gray, k, dR);
//	cv::imshow("sauvola", mat2gray(sauvola));
//
//	cv::Mat wolf = binWolf(gray, k, dR);
//	cv::imshow("wolf", mat2gray(wolf));
//
	cv::Mat result = gcnlBin(image, "n");
	cv::imshow("result", mat2gray(result));

	cv::waitKey(0);

    return 0;
}

