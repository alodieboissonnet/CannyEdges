#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

// Step 1: complete gradient and threshold
// Step 2: complete sobel
// Step 3: complete canny (recommended substep: return Max instead of C to check it)

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

	for (int i = 1; i < m-1; i++) {
		for (int j = 1; j < n-1; j++) {
			G2.at<float>(i, j) = pow((I.at<uchar>(i+1,j) - I.at<uchar>(i-1,j))/2,2) + pow((I.at<uchar>(i,j+1) - I.at<uchar>(i,j-1))/2,2);
		}
	}
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	float values_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float values_y[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	Mat sobel_x(3, 3, CV_32F, values_x);
	Mat sobel_y(3, 3, CV_32F, values_y);

	Mat current(3, 3, CV_32F);
	for (int i = 1; i < m-1; i++) {
		for (int j = 1; j < n-1; j++) {
			Mat(I, Rect(j-1, i-1, 3, 3)).convertTo(current, CV_32F);
			Ix.at<float>(i, j) = sum(Mat(current.mul(sobel_x)))[0]/255.;
			Iy.at<float>(i, j) = sum(Mat(current.mul(sobel_y)))[0]/255.;
			G2.at<float>(i, j) = pow(Ix.at<float>(i, j), 2) + pow(Iy.at<float>(i, j), 2);
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++){
			C.at<uchar>(i, j) = G2.at<float>(i, j) > s ? 255 : 0;
	}
	return C;
}


int sgn(float val) {
		if (val > 0)
			return 1;
		else if (val < 0)
			return -1;
		else
			return 0;
}

// Canny edge detector
Mat canny(const Mat& Ic, float s1, float s2)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Max pixels ( G2 > s1 && max in the direction of the gradient )
	Max.setTo(0);
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
	for (int i = 1; i < m-1; i++) {
		for (int j = 1; j < n-1; j++) {
			// find value in the direction of the gradient
			float value;
			float Gx = Ix.at<float>(i, j);
			float Gy = Iy.at<float>(i, j);
			if (abs(Gx) > 2.5 * abs(Gy))
				value = G2.at<float>(i, j + sgn(Gx));
			else if (abs(Gy) > 2.5 * abs(Gx))
				value = G2.at<float>(i + sgn(Gy), j);
			else
				value = G2.at<float>(i + sgn(Gy), j + sgn(Gx));

			if (G2.at<float>(i, j) > s2)
				Q.push(Point(j, i)); // Beware: Mats use row,col, but points use x,y

			if (G2.at<float>(i, j) > s1 && G2.at<float>(i, j) > value)
				Max.at<uchar>(i, j) = 255;
		}
	}

	// Propagate seeds
	Mat C(m, n, CV_8U);
	C.setTo(0);
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();

		for (int k = -1; k < 2; k++) {
			for (int l = -1; l < 2; l++) {
				if (C.at<uchar>(i + k, j + l) == 0 && Max.at<uchar>(i + k, j + l) != 0) {
						C.at<uchar>(i + k, j + l) = Max.at<uchar>(i + k, j + l);
						Q.push(Point(j + l, i + k));
				}
			}
		}
	}
	return C;
}

int main()
{
	Mat I = imread("../road.jpg");

	imshow("Input", I);
	imshow("Threshold", threshold(I, 1));
	imshow("Threshold + denoising", threshold(I, 1, true));
	imshow("Canny", canny(I, 1, 2));

	waitKey();

	return 0;
}
