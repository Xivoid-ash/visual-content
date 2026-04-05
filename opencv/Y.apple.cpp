#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // 1. 读取图像
    Mat src = imread("Resources/apple.png"); // 替换为你的图片路径
    if (src.empty()) {
        cout << "无法加载图像！" << endl;
        return -1;
    }

    // 2. 转换到HSV颜色空间（精准分割红色苹果）
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 3. 定义红色的HSV阈值范围（覆盖苹果红色区域）
    Scalar lower_red1 = Scalar(0, 173, 90);
    Scalar upper_red1 = Scalar(26, 255, 255);
    Scalar lower_red2 = Scalar(157, 173, 90);
    Scalar upper_red2 = Scalar(180, 255, 255);

    // 4. 提取红色掩码（合并两个红色区间）
    Mat mask1, mask2, mask;
    inRange(hsv, lower_red1, upper_red1, mask1);
    inRange(hsv, lower_red2, upper_red2, mask2);
    mask = mask1 | mask2;

    // 5. 形态学操作优化掩码（填充孔洞、去除噪点）
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel); // 闭运算填充内部孔洞
    morphologyEx(mask, mask, MORPH_OPEN, kernel);  // 开运算去除小噪点

    // 6. 轮廓检测与过滤
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double max_area = 0;
    int max_idx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area && area > 1000) { // 过滤微小轮廓
            max_area = area;
            max_idx = i;
        }
    }

    // 7. 绘制结果（苹果轮廓 + 框住外接圆的矩形 + 最大外接圆）
    Mat result = src.clone();
    if (max_idx != -1) {
        // 轮廓近似（减少噪点，让圆形拟合更精准）
        vector<Point> approx_contour;
        approxPolyDP(contours[max_idx], approx_contour, arcLength(contours[max_idx], true) * 0.01, true);

        // 绘制苹果轮廓（绿色）
        drawContours(result, contours, max_idx, Scalar(0, 255, 0), 2);

        // 拟合最大外接圆（蓝色，核心需求）
        Point2f center; // 圆心
        float radius;   // 半径
        minEnclosingCircle(approx_contour, center, radius);
        circle(result, center, (int)radius, Scalar(255, 0, 0), 2); // 蓝色圆形框

        // 计算能框住外接圆的矩形（核心修改点）
        int rect_x = static_cast<int>(center.x - radius);  // 矩形左上角x
        int rect_y = static_cast<int>(center.y - radius);  // 矩形左上角y
        int rect_width = static_cast<int>(2 * radius);     // 矩形宽度（直径）
        int rect_height = static_cast<int>(2 * radius);    // 矩形高度（直径）
        Rect circle_rect(rect_x, rect_y, rect_width, rect_height);

        // 绘制框住外接圆的红色矩形
        rectangle(result, circle_rect, Scalar(0, 0, 255), 2);

        // 输出圆形框和矩形框信息
        cout << "苹果最大外接圆信息：" << endl;
        cout << "圆心坐标：(" << center.x << ", " << center.y << ")" << endl;
        cout << "半径：" << radius << endl;
        cout << "框住圆的矩形：" << endl;
        cout << "左上角：(" << rect_x << ", " << rect_y << ")" << endl;
        cout << "宽高：" << rect_width << " x " << rect_height << endl;
    }

    // 8. 显示结果
    imshow("原始图像", src);
    imshow("苹果掩码", mask);
    imshow("苹果轮廓 + 框圆矩形 + 最大圆形框", result);

    // 保存结果
    imwrite("apple_circle_result.jpg", result);

    waitKey(0);
    destroyAllWindows();
    return 0;
}