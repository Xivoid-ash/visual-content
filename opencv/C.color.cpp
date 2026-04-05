#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

Mat imgHSV , mask;
int hmin = 124, smin = 35, vmin = 153;
int hmax = 179, smax = 255, vmax = 255;
//(32, 100, 39, 127, -95, 127)

int main()
{
	string path = "Resources/color.png";
	Mat img = imread(path);


	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	namedWindow("Trackbars", (640, 480));
	createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);

	imshow("Imagehsv", imgHSV);
    Mat img_resized, mask_resized, edge_resized;
	while (true) {

		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		inRange(imgHSV, lower, upper, mask);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
		morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);
		morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);
		Mat edges;
		Canny(mask, edges, 50, 150);
		Mat edge_color = Mat::zeros(img.size(), img.type());
		edge_color.setTo(Scalar(255, 255, 0), edges);

		Size target_size(1280, 720);
		
		resize(img, img_resized, target_size);
		resize(mask, mask_resized, target_size);
		resize(edge_color, edge_resized, target_size);


		imshow("Original Image", img_resized);
		imshow("Red Binary Mask", mask_resized);
		imshow("Edge Detection (Light Blue)", edge_resized);

		char key = waitKey(1);
		if (key == 'q') {
			break;
		}
	}
	bool save_success = imwrite("edge_result.png",edge_resized);
	if (save_success) {
		cout << "±ßÔµÍ¼ÏñÒÑ³É¹¦±£´æÎª edge_result.png" << endl;
	}
	else {
		cout << "±£´æ±ßÔµÍ¼ÏñÊ§°Ü£¡" << endl;
	}
	destroyAllWindows();
	return 0;



}