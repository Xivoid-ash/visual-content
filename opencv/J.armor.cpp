#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;

// ================== 装甲板识别参数（优化版） ==================
// 灯条筛选参数
const int LIGHT_MIN_AREA = 50;          // 灯条最小面积（降低以适应远距离）
const int LIGHT_MAX_AREA = 50000;       // 灯条最大面积（提高以适应近距离）
const float LIGHT_MIN_RATIO = 2.0f;     // 灯条最小长宽比
const float LIGHT_MAX_RATIO = 10.0f;    // 灯条最大长宽比（降低以排除误检）
const float LIGHT_MAX_ANGLE = 30.0f;    // 灯条最大倾斜角度（收窄）
const float LIGHT_CONVEX_RATIO = 0.6f;  // 灯条凸包面积比（排除不规则形状）

// 装甲板匹配参数
const float ARMOR_MAX_ANGLE_DIFF = 8.0f;    // 两灯条最大角度差
const float ARMOR_MAX_HEIGHT_RATIO = 0.25f; // 两灯条最大高度差比例（收窄）
const float ARMOR_MIN_WIDTH_RATIO = 1.5f;   // 两灯条中心距/灯条高度的最小比例
const float ARMOR_MAX_WIDTH_RATIO = 3.5f;   // 两灯条中心距/灯条高度的最大比例（收窄）
const float ARMOR_ASPECT_RATIO_MIN = 1.5f;  // 装甲板最小宽高比
const float ARMOR_ASPECT_RATIO_MAX = 4.0f;  // 装甲板最大宽高比

// 帧间跟踪参数
const float ROI_EXPAND_RATIO = 1.5f;         // ROI区域扩展比例
bool last_frame_detected = false;             // 上一帧是否检测到装甲板
Rect last_armor_roi;                          // 上一帧装甲板的ROI

// ================== 灯条结构体 ==================
struct LightBar {
    RotatedRect rect;
    float angle;
    float height;
    float width;
    Point2f center;
    float area;

    LightBar(RotatedRect r, float a) : rect(r), area(a) {
        center = r.center;
        if (r.size.width > r.size.height) {
            angle = r.angle - 90.0f;
            height = r.size.width;
            width = r.size.height;
        }
        else {
            angle = r.angle;
            height = r.size.height;
            width = r.size.width;
        }
    }
};

// ================== 图像预处理：HSV颜色空间过滤 ==================
Mat preprocessImage(const Mat& frame, bool is_blue = true) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat mask;
    if (is_blue) {
        // 蓝色HSV范围（RoboMaster场景通用）
        Scalar lower_blue(78, 0, 210);
        Scalar upper_blue(113, 255, 255);
        inRange(hsv, lower_blue, upper_blue, mask);
    }
    else {
        // 红色HSV范围（两个区间，核心解决红色无法识别问题！）
        Mat mask1, mask2;
        Scalar lower_red1(0, 208, 76);
        Scalar upper_red1(38, 255, 255);
        Scalar lower_red2(172, 208, 76);
        Scalar upper_red2(180, 255, 255);
        inRange(hsv, lower_red1, upper_red1, mask1);
        inRange(hsv, lower_red2, upper_red2, mask2);
        mask = mask1 | mask2;
    }

    // 形态学操作：先开运算去噪，再闭运算填充缺口
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    return mask;
}

// ================== 检测并筛选灯条（增加凸包约束） ==================
vector<LightBar> detectLightBars(const Mat& binary, const Mat& roi_mask = Mat()) {
    vector<vector<Point>> contours;
    if (roi_mask.empty()) {
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    }
    else {
        Mat roi_binary = binary & roi_mask;
        findContours(roi_binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    }

    vector<LightBar> light_bars;
    for (const auto& contour : contours) {
        float area = contourArea(contour);
        // 1. 面积筛选
        if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

        // 2. 凸包面积比筛选（排除不规则误检）
        vector<Point> hull;
        convexHull(contour, hull);
        float hull_area = contourArea(hull);
        if (area / hull_area < LIGHT_CONVEX_RATIO) continue;

        RotatedRect rect = minAreaRect(contour);
        LightBar light(rect, area);

        // 3. 长宽比筛选
        float ratio = light.height / light.width;
        if (ratio < LIGHT_MIN_RATIO || ratio > LIGHT_MAX_RATIO) continue;

        // 4. 角度筛选
        if (abs(light.angle) > LIGHT_MAX_ANGLE) continue;

        light_bars.push_back(light);
    }
    return light_bars;
}

// ================== 匹配灯条对，识别装甲板（增加装甲板宽高比约束） ==================
vector<pair<LightBar, LightBar>> matchArmors(const vector<LightBar>& light_bars) {
    vector<pair<LightBar, LightBar>> armors;
    int n = light_bars.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const LightBar& l1 = light_bars[i], & l2 = light_bars[j];

            // 1. 角度差筛选
            if (abs(l1.angle - l2.angle) > ARMOR_MAX_ANGLE_DIFF) continue;

            // 2. 高度差筛选
            float h_diff = abs(l1.height - l2.height), h_max = max(l1.height, l2.height);
            if (h_diff / h_max > ARMOR_MAX_HEIGHT_RATIO) continue;

            // 3. 中心距离筛选
            float c_dist = norm(l1.center - l2.center), h_avg = (l1.height + l2.height) / 2.0f;
            if (c_dist / h_avg < ARMOR_MIN_WIDTH_RATIO || c_dist / h_avg > ARMOR_MAX_WIDTH_RATIO) continue;

            // 4. 装甲板宽高比筛选
            vector<Point2f> all_points;
            Point2f p1[4], p2[4];
            l1.rect.points(p1);
            l2.rect.points(p2);
            for (int k = 0; k < 4; k++) { all_points.push_back(p1[k]); all_points.push_back(p2[k]); }
            Rect armor_rect = boundingRect(all_points);
            float armor_aspect = (float)armor_rect.width / armor_rect.height;
            if (armor_aspect < ARMOR_ASPECT_RATIO_MIN || armor_aspect > ARMOR_ASPECT_RATIO_MAX) continue;

            armors.emplace_back(l1, l2);
        }
    }
    return armors;
}

// ================== 绘制结果（含帧间跟踪ROI） ==================
void drawResults(Mat& frame, const vector<pair<LightBar, LightBar>>& armors, float process_time) {
    // 绘制上一帧的ROI（可选，用于调试）
    if (last_frame_detected) {
        rectangle(frame, last_armor_roi, Scalar(255, 255, 0), 1);
    }

    for (const auto& armor : armors) {
        // 绘制灯条
        Point2f p1[4], p2[4];
        armor.first.rect.points(p1);
        armor.second.rect.points(p2);
        for (int i = 0; i < 4; i++) {
            line(frame, p1[i], p1[(i + 1) % 4], Scalar(0, 255, 0), 2);
            line(frame, p2[i], p2[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }

        // 绘制装甲板中心和外接矩形
        Point2f center = (armor.first.center + armor.second.center) / 2.0f;
        circle(frame, center, 5, Scalar(0, 0, 255), -1);
        vector<Point2f> all_points;
        for (int i = 0; i < 4; i++) { all_points.push_back(p1[i]); all_points.push_back(p2[i]); }
        Rect rect = boundingRect(all_points);
        rectangle(frame, rect, Scalar(255, 0, 0), 2);

        // 更新帧间跟踪ROI
        last_armor_roi = Rect(
            max(0, (int)(rect.x - rect.width * (ROI_EXPAND_RATIO - 1) / 2)),
            max(0, (int)(rect.y - rect.height * (ROI_EXPAND_RATIO - 1) / 2)),
            min(frame.cols, (int)(rect.width * ROI_EXPAND_RATIO)),
            min(frame.rows, (int)(rect.height * ROI_EXPAND_RATIO))
        );
        last_frame_detected = true;
    }

    // 无检测时重置跟踪
    if (armors.empty()) {
        last_frame_detected = false;
    }

    putText(frame, "Time: " + to_string(process_time) + " ms", Point(10, 30),
        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
}

int main() {
    string video_path = "Resources/armor_video_red.mp4"; // 替换为你的视频路径
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "无法打开视频：" << video_path << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = chrono::high_resolution_clock::now();

        // 1. 图像预处理（HSV颜色过滤，切换is_blue=false识别红色）
        Mat binary = preprocessImage(frame, false);

        // 2. 检测灯条（带帧间ROI跟踪）
        Mat roi_mask = Mat::zeros(frame.size(), CV_8UC1);
        if (last_frame_detected) {
            roi_mask(last_armor_roi).setTo(255);
        }
        vector<LightBar> lights = detectLightBars(binary, roi_mask);

        // 3. 匹配装甲板
        auto armors = matchArmors(lights);

        auto end = chrono::high_resolution_clock::now();
        float time_ms = chrono::duration<float, milli>(end - start).count();

        drawResults(frame, armors, time_ms);
        imshow("Optimized Armor Detection", frame);
        imshow("Optimized Armor HSV", binary);
        int key = waitKey(30);
        if (key == 27) break;
        if (key == 32) waitKey(0);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}