#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat img;
Point start_pt, end_pt;
bool is_dragging = false;

// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        start_pt = Point(x, y);
        end_pt = Point(x, y);
        is_dragging = true;
    }
    else if (event == EVENT_MOUSEMOVE && is_dragging) {
        end_pt = Point(x, y);
        // 显示当前鼠标位置和RGB值
        Vec3b pixel = img.at<Vec3b>(y, x);
        cout << "坐标: (" << x << ", " << y << "), RGB: ("
            << (int)pixel[2] << ", " << (int)pixel[1] << ", " << (int)pixel[0] << ")\r";
        cout.flush();
    }
    else if (event == EVENT_LBUTTONUP) {
        end_pt = Point(x, y);
        is_dragging = false;

        // 确保选区是正矩形
        int x_min = min(start_pt.x, end_pt.x);
        int x_max = max(start_pt.x, end_pt.x);
        int y_min = min(start_pt.y, end_pt.y);
        int y_max = max(start_pt.y, end_pt.y);
        Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);

        // 裁剪并保存选区
        Mat cropped = img(roi);
        imwrite("cropped_cat.jpg", cropped);
        imshow("Cropped Cat", cropped);

        // 计算并输出中心坐标
        Point center((x_min + x_max) / 2, (y_min + y_max) / 2);
        cout << "\n框选中心坐标: (" << center.x << ", " << center.y << ")" << endl;
    }
}

int main() {
    img = imread("Resources/cat.png"); 
    if (img.empty()) {
        cout << "无法加载图片" << endl;
        return -1;
    }

    namedWindow("Cat Image");
    setMouseCallback("Cat Image", onMouse);

    while (true) {
        Mat temp = img.clone();
        if (is_dragging) {
            rectangle(temp, start_pt, end_pt, Scalar(0, 255, 0), 2);
        }
        imshow("Cat Image", temp);
        if (waitKey(1) == 'q') break;
    }
    destroyAllWindows();
    return 0;
}