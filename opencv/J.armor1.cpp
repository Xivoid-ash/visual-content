#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

const int LIGHT_MIN_AREA = 20;
const int LIGHT_MAX_AREA = 2000;
const float LIGHT_MIN_RATIO = 2.0f;
const float LIGHT_MAX_RATIO = 10.0f;
const float LIGHT_MAX_ANGLE = 45.0f;
const float LIGHT_MAX_HEIGHT = 150;

const float ARMOR_MAX_ANGLE_DIFF = 10.0f;
const float ARMOR_MAX_HEIGHT_RATIO = 0.3f;
const float ARMOR_MIN_WIDTH_RATIO = 1.0f;
const float ARMOR_MAX_WIDTH_RATIO = 4.0f;


// 左右灯条最外沿距离：140mm
// 单个灯条自身高度：100mm
const vector<Point3f> ARMOR_3D_POINTS = {
    Point3f(-70.0f,  50.0f, 0.0f),  // 0: 左灯条最顶部
    Point3f(-70.0f, -50.0f, 0.0f),  // 1: 左灯条最底部
    Point3f(70.0f,   50.0f, 0.0f),  // 2: 右灯条最顶部
    Point3f(70.0f,  -50.0f, 0.0f)   // 3: 右灯条最底部
};

//相机内参
const Mat CAMERA_MATRIX = (Mat_<double>(3, 3) <<
    550.0f, 0.0f, 640.0f,
    0.0f, 550.0f, 360.0f,
    0.0f, 0.0f, 1.0f);
const Mat DIST_COEFFS = (Mat_<double>(1, 5) << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

// 灯条结构体
struct LightBar {
    RotatedRect rect;
    float angle;
    float height;
    Point2f center;
    vector<Point2f> corners;

    LightBar(RotatedRect r) : rect(r) {
        center = r.center;
        if (r.size.width > r.size.height) {
            angle = r.angle - 90.0f;
            height = r.size.width;
        }
        else {
            angle = r.angle;
            height = r.size.height;
        }
        Point2f pts[4];
        r.points(pts);
        corners.assign(pts, pts + 4);
    }
};

//图像预处理
Mat preprocessImage(const Mat& frame) {
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    Scalar lower_blue(78, 0, 160);
    Scalar upper_blue(113, 255, 255);
    inRange(hsv, lower_blue, upper_blue, mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    return mask;
}

//检测灯条
vector<LightBar> detectLightBars(const Mat& binary) {
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<LightBar> light_bars;
    for (const auto& contour : contours) {
        float area = contourArea(contour);
        if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

        RotatedRect rect = minAreaRect(contour);
        LightBar light(rect);

        if (light.height > LIGHT_MAX_HEIGHT) continue;

        float ratio = light.height / min(rect.size.width, rect.size.height);
        if (ratio < LIGHT_MIN_RATIO || ratio > LIGHT_MAX_RATIO) continue;
        if (abs(light.angle) > LIGHT_MAX_ANGLE) continue;

        light_bars.push_back(light);
    }
    return light_bars;
}

//匹配装甲板
vector<pair<LightBar, LightBar>> matchArmors(const vector<LightBar>& light_bars) {
    vector<pair<LightBar, LightBar>> armors;
    int n = light_bars.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const LightBar& l1 = light_bars[i], & l2 = light_bars[j];
            if (abs(l1.angle - l2.angle) > ARMOR_MAX_ANGLE_DIFF) continue;

            float h_diff = abs(l1.height - l2.height);
            float h_max = max(l1.height, l2.height);
            if (h_max < 1e-6) continue;
            if (h_diff / h_max > ARMOR_MAX_HEIGHT_RATIO) continue;

            float c_dist = norm(l1.center - l2.center);
            float h_avg = (l1.height + l2.height) / 2.0f;
            if (h_avg < 1e-6) continue;
            if (c_dist / h_avg < ARMOR_MIN_WIDTH_RATIO || c_dist / h_avg > ARMOR_MAX_WIDTH_RATIO) continue;

            armors.emplace_back(l1, l2);
            return armors;
        }
    }
    return armors;
}

//PnP解算距离
float calculateDistance(const pair<LightBar, LightBar>& armor) {
    if (armor.first.corners.size() != 4 || armor.second.corners.size() != 4) return -1.0f;

    // 左右
    LightBar left_bar = armor.first.center.x < armor.second.center.x ? armor.first : armor.second;
    LightBar right_bar = armor.first.center.x < armor.second.center.x ? armor.second : armor.first;

    // 按Y坐标排序
    vector<Point2f> left_corners = left_bar.corners;
    vector<Point2f> right_corners = right_bar.corners;
    sort(left_corners.begin(), left_corners.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
    sort(right_corners.begin(), right_corners.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

    if (left_corners.size() < 4 || right_corners.size() < 4) return -1.0f;

    // 左上、左下、右上、右下
    vector<Point2f> image_points;
    image_points.push_back(left_corners[0]);  
    image_points.push_back(left_corners[3]);  
    image_points.push_back(right_corners[0]); 
    image_points.push_back(right_corners[3]);

    // 解算
    Mat rvec, tvec;
    try {
        solvePnP(ARMOR_3D_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec);
        if (tvec.empty()) return -1.0f;
        return (float)norm(tvec);
    }
    catch (...) {
        return -1.0f;
    }
}

//截取ROI，曝光
Mat cropAndBrightenROI(const Mat& frame, const pair<LightBar, LightBar>& armor) {
    if (frame.empty()) return Mat();
    if (armor.first.corners.size() != 4 || armor.second.corners.size() != 4) return Mat();

    vector<Point2f> all_points;
    for (const auto& p : armor.first.corners) all_points.push_back(p);
    for (const auto& p : armor.second.corners) all_points.push_back(p);
    Rect roi_rect = boundingRect(all_points);

    // 扩展ROI以包住数字
    float avg_light_height = (armor.first.height + armor.second.height) / 2.0f;
    int expand_pixels = static_cast<int>(avg_light_height * 0.8f);
    roi_rect.x = max(0, roi_rect.x - expand_pixels);
    roi_rect.y = max(0, roi_rect.y - expand_pixels);
    roi_rect.width = min(frame.cols - roi_rect.x, roi_rect.width + 2 * expand_pixels);
    roi_rect.height = min(frame.rows - roi_rect.y, roi_rect.height + 2 * expand_pixels);

    Rect image_bound(0, 0, frame.cols, frame.rows);
    roi_rect = roi_rect & image_bound;

    if (roi_rect.width <= 0 || roi_rect.height <= 0) return Mat();

    Mat roi = frame(roi_rect).clone();
    Mat bright_roi;
    roi.convertTo(bright_roi, -1, 1.5, 50); 
    return bright_roi;
}

int main() {
    string img_path = "Resources/armor.png";
    Mat frame = imread(img_path);
    if (frame.empty()) { cerr << "无法读取图片" << endl; return -1; }
    Mat result_frame = frame.clone();

    //延迟计时开始
    auto t_start = high_resolution_clock::now();

    Mat binary = preprocessImage(frame);
    vector<LightBar> lights = detectLightBars(binary);
    auto armors = matchArmors(lights);

    // 只处理第一个装甲板
    if (!armors.empty()) {
        const auto& armor = armors[0];

        // 绘制灯条
        Point2f p1[4], p2[4];
        armor.first.rect.points(p1);
        armor.second.rect.points(p2);
        for (int j = 0; j < 4; j++) {
            line(result_frame, p1[j], p1[(j + 1) % 4], Scalar(0, 255, 0), 2);
            line(result_frame, p2[j], p2[(j + 1) % 4], Scalar(0, 255, 0), 2);
        }
        vector<Point2f> all_pts;
        for (int j = 0; j < 4; j++) { all_pts.push_back(p1[j]); all_pts.push_back(p2[j]); }
        Rect armor_rect = boundingRect(all_pts);
        rectangle(result_frame, armor_rect, Scalar(255, 0, 0), 2);

        // 绘制中心点
        Point2f center = (armor.first.center + armor.second.center) / 2.0f;
        circle(result_frame, center, 3, Scalar(0, 0, 255), -1);

        // 计算距离
        float dist_mm = calculateDistance(armor);
        if (dist_mm > 0) {
            float dist_cm = dist_mm / 10.0f;
            cout << "装甲板 距离: " << dist_cm << " cm" << endl;
        }
        //总延迟计时结束
        auto t_end = high_resolution_clock::now();
        float total_time_ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0f;
        cout << "------------------------" << endl;
        cout << ">> 总处理延迟: " << total_time_ms << " ms" << endl;
        cout << "------------------------" << endl;


        // 截取ROI
        Mat bright_roi = cropAndBrightenROI(frame, armor);
        if (!bright_roi.empty()) {
            imshow("ROI", bright_roi);
        }
    }

    imshow("Binary", binary);
    imshow("Result", result_frame);
    waitKey(0);
    return 0;
}