#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

//装甲板识别参数
const int LIGHT_MIN_AREA = 300;
const int LIGHT_MAX_AREA = 100000;
const float LIGHT_MIN_RATIO = 2.0f;
const float LIGHT_MAX_RATIO = 15.0f;
const float LIGHT_MAX_ANGLE = 45.0f;

const float ARMOR_MAX_ANGLE_DIFF = 8.0f;
const float ARMOR_MAX_HEIGHT_RATIO = 0.3f;
const float ARMOR_MAX_WIDTH_RATIO = 4.0f;

//相机标定参数
const Mat CAMERA_MATRIX = (Mat_<double>(3, 3) <<
    550.0f, 0.0f, 640.0f,
    0.0f, 550.0f, 360.0f,
    0.0f, 0.0f, 1.0f);
const Mat DIST_COEFFS = (Mat_<double>(1, 5) << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

// 装甲板3D坐标（235mm × 110mm）
const vector<Point3f> ARMOR_3D_POINTS = {
    Point3f(-117.5f,  55.0f, 0.0f),  // 左上
    Point3f(-117.5f, -55.0f, 0.0f),  // 左下
    Point3f(117.5f,   55.0f, 0.0f),  // 右上
    Point3f(117.5f,  -55.0f, 0.0f)   // 右下
};

// 灯条结构体
struct LightBar {
    RotatedRect rect;
    float angle;
    float height;
    Point2f center;
    vector<Point2f> corners;  

    LightBar(RotatedRect r) {
        rect = r;
        center = r.center;
        if (r.size.width > r.size.height) {
            angle = r.angle - 90.0f;
            height = r.size.width;
        }
        else {
            angle = r.angle;
            height = r.size.height;
        }
        // 获取4个角点
        Point2f pts[4];
        r.points(pts);
        corners.assign(pts, pts + 4);
    }
};

//HSV 预处理
Mat preprocessImage(const Mat& frame, bool is_blue = true) {
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    if (is_blue) {
        Scalar lower(78, 0, 210);
        Scalar upper(113, 255, 255);
        inRange(hsv, lower, upper, mask);
    }
    else {
        Mat mask1, mask2;
        inRange(hsv, Scalar(0, 193, 157), Scalar(32, 255, 255), mask1);
        inRange(hsv, Scalar(144, 193, 157), Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

// 检测灯条
vector<LightBar> detectLightBars(const Mat& binary) {
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<LightBar> light_bars;
    for (const auto& contour : contours) {
        float area = contourArea(contour);
        if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

        RotatedRect rect = minAreaRect(contour);
        LightBar light(rect);

        float ratio = light.height / min(rect.size.width, rect.size.height);
        if (ratio < LIGHT_MIN_RATIO || ratio > LIGHT_MAX_RATIO) continue;
        if (abs(light.angle) > LIGHT_MAX_ANGLE) continue;

        light_bars.push_back(light);
    }
    return light_bars;
}

// 匹配装甲板
vector<pair<LightBar, LightBar>> matchArmors(const vector<LightBar>& light_bars) {
    vector<pair<LightBar, LightBar>> armors;
    int n = light_bars.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const LightBar& light1 = light_bars[i];
            const LightBar& light2 = light_bars[j];

            if (abs(light1.angle - light2.angle) > ARMOR_MAX_ANGLE_DIFF) continue;

            float height_diff = abs(light1.height - light2.height);
            float max_height = max(light1.height, light2.height);
            if (height_diff / max_height > ARMOR_MAX_HEIGHT_RATIO) continue;

            float center_dist = norm(light1.center - light2.center);
            float avg_height = (light1.height + light2.height) / 2.0f;
            if (center_dist / avg_height > ARMOR_MAX_WIDTH_RATIO) continue;

            armors.emplace_back(light1, light2);
            return armors; 
        }
    }
    return armors;
}

// PnP解算距离
float calculateDistance(const pair<LightBar, LightBar>& armor) {
    if (armor.first.corners.size() != 4 || armor.second.corners.size() != 4) return -1.0f;

    LightBar left_bar = armor.first.center.x < armor.second.center.x ? armor.first : armor.second;
    LightBar right_bar = armor.first.center.x < armor.second.center.x ? armor.second : armor.first;

    vector<Point2f> left_corners = left_bar.corners;
    vector<Point2f> right_corners = right_bar.corners;
    sort(left_corners.begin(), left_corners.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
    sort(right_corners.begin(), right_corners.end(), [](const Point2f& a, const Point2f& b) { return a.y < b.y; });

    vector<Point2f> image_points;
    image_points.push_back(left_corners[0]);
    image_points.push_back(left_corners[3]);
    image_points.push_back(right_corners[0]);
    image_points.push_back(right_corners[3]);

    Mat rvec, tvec;
    try {
        solvePnP(ARMOR_3D_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec);
        return (float)norm(tvec); // 返回毫米
    }
    catch (...) {
        return -1.0f;
    }
}

//  绘制结果
void drawResults(Mat& frame, const pair<LightBar, LightBar>& armor) {
    Point2f points1[4], points2[4];
    armor.first.rect.points(points1);
    armor.second.rect.points(points2);
    for (int i = 0; i < 4; i++) {
        line(frame, points1[i], points1[(i + 1) % 4], Scalar(0, 255, 0), 2);
        line(frame, points2[i], points2[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }

    Point2f armor_center = (armor.first.center + armor.second.center) / 2.0f;
    circle(frame, armor_center, 5, Scalar(0, 0, 255), -1);

    vector<Point2f> all_points;
    for (int i = 0; i < 4; i++) {
        all_points.push_back(points1[i]);
        all_points.push_back(points2[i]);
    }
    Rect armor_rect = boundingRect(all_points);
    rectangle(frame, armor_rect, Scalar(255, 0, 0), 2);
}

// 主函数
int main() {
    string video_path = "Resources/armor_video_bule.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件：" << video_path << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "视频播放完毕！" << endl;
            break;
        }

        // 计时开始
        auto start = high_resolution_clock::now();

        // 核心识别流程
        Mat binary = preprocessImage(frame, true);
        vector<LightBar> lights = detectLightBars(binary);
        auto armors = matchArmors(lights);

        // 计时结束
        auto end = high_resolution_clock::now();
        float time_ms = duration<float, milli>(end - start).count();

        // 仅处理第一个装甲板
        if (!armors.empty()) {
            const auto& armor = armors[0];
            drawResults(frame, armor);

            // 解算距离
            float dist_mm = calculateDistance(armor);
            if (dist_mm > 0) {
                float dist_cm = dist_mm / 10.0f;
                cout << "装甲板距离: " << dist_cm << " cm | 处理延迟: " << time_ms << " ms" << endl;
            }
        }

        imshow("Armor Detection (Video)", frame);

        int key = waitKey(1); 
        if (key == 27) break;
        if (key == 32) waitKey(0);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}