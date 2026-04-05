#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>


using namespace cv;
using namespace std;

Mat imgHSV, mask;
// 定义HSV阈值
int hmin = 124, smin = 35, vmin = 153;
int hmax = 179, smax = 255, vmax = 255;

int main()
{
	string path = "Resources/color.png";
	Mat img = imread(path);

	//转换为HSV
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	namedWindow("Trackbars", (640, 480));
	// 创建6个滑动条，分别控制H、S、V的最小值和最大值
	createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);

	// 显示HSV图像
	imshow("Imagehsv", imgHSV);

	Mat img_resized, mask_resized, edge_resized;

	// 循环实时更新阈值效果
	while (true) {
		// 根据滑动条值定义HSV下限和上限
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);

		// 生成二值掩码
		inRange(imgHSV, lower, upper, mask);

		// 形态学操作
		Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
		morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);
		morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);

		// Canny边缘
		Mat edges;
		Canny(mask, edges, 50, 150);

		// 创建黑色画布
		Mat edge_color = Mat::zeros(img.size(), img.type());
		// 将边缘位置设为蓝色
		edge_color.setTo(Scalar(255, 255, 0), edges);

		Size target_size(1280, 720);
		resize(img, img_resized, target_size);
		resize(mask, mask_resized, target_size);
		resize(edge_color, edge_resized, target_size);

		// 显示原图、掩码、彩色边缘图
		imshow("Original Image", img_resized);
		imshow("Red Binary Mask", mask_resized);
		imshow("Edge Detection (Light Blue)", edge_resized);

		char key = waitKey(1);
		if (key == 'q') {
			break;
		}
	}

	// 保存最终的彩色边缘图像
	bool save_success = imwrite("edge_result.png", edge_resized);
	// 输出保存结果
	if (save_success) {
		cout << "边缘图像已成功保存为 edge_result.png" << endl;
	}
	else {
		cout << "保存边缘图像失败！" << endl;
	}
	destroyAllWindows();
	return 0;
}