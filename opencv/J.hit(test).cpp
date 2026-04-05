#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
using namespace std::chrono;

//配置参数
struct Config {
    Scalar hsv_low = Scalar(0, 120, 180);
    Scalar hsv_high = Scalar(25, 255, 255);
    int dilate_kernel = 2;
    float r_size_err = 0.1f;
    int max_lost = 8;
    int roi_expand = 40;
    float approx_epsilon = 0.008;
    float min_fan_dist = 50.0f;
    float max_fan_dist = 300.0f;
    float min_fan_area = 100;
    float max_fan_area = 10000;
    float broad_radius = 600;
    float broad_inner = 40;
    int state2_min_duration = 40;
};

// 状态机定义
enum TrackPhase {
    FIND_R,
    FIND_FAN,
    LOCK_RADIUS,
    TRACK
};

enum BladeState { UNDEF, TARGET, HIT };

struct Box {
    Rect rect;
    Point2f center;
    float area = 0;
    int id = -1;

    Box() {}
    Box(Rect r) : rect(r) {
        center = Point2f(r.x + r.width / 2.f, r.y + r.height / 2.f);
        area = r.area();
    }
};

pair<Point2f, float> fitCircle(const vector<Point>& pts) {
    if (pts.size() < 5) return { Point2f(-1,-1), -1.0f };
    double sumx = 0, sumy = 0, sumx2 = 0, sumy2 = 0, sumxy = 0;
    for (auto& p : pts) { sumx += p.x; sumy += p.y; sumx2 += p.x * p.x; sumy2 += p.y * p.y; sumxy += p.x * p.y; }
    int n = pts.size();
    double A = n * sumx2 - sumx * sumx;
    double B = n * sumxy - sumx * sumy;
    double C = n * sumy2 - sumy * sumy;
    double D = 0.5 * ((sumx2 + sumy2) * n - (sumx * sumx + sumy * sumy));
    double cx = (B * D - sumy * A) / (A * C - B * B);
    double cy = (sumx * C - B * D) / (A * C - B * B);
    double r = sqrt((sumx2 + sumy2 - sumx * cx - sumy * cy) / n - cx * cx - cy * cy);
    return { Point2f(cx,cy), r };
}

class EnergyTracker {
public:
    Config cfg;
    Box R;
    Rect fixed_R_rect;
    Point2f fixed_R_center;
    bool is_R_fixed = false;

    Box fans[5];
    BladeState fan_state[5];
    float outer_r = 0, inner_r = 0, avg_r = 0;
    int lost = 0;
    bool r_found = false, radius_locked = false;
    float fan_w = 0, fan_h = 0;
    float angle = 0;
    TrackPhase phase;

    steady_clock::time_point state2_start_time;
    bool state2_timer_started = false;

    // 状态二专用：保存闭合轮廓 + 拟合圆参数
    vector<Point> closed_fan_contour;
    Point2f circle_center;
    float circle_radius;

    EnergyTracker() : phase(FIND_R) {
        for (int i = 0; i < 5; i++) fan_state[i] = UNDEF;
        // 初始化变量
        closed_fan_contour.clear();
        circle_center = Point2f(0, 0);
        circle_radius = 0;
    }

    bool findR(const Mat& dil) {
        vector<vector<Point>> contours;
        findContours(dil, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& c : contours) {
            Rect r = boundingRect(c);
            if (abs(r.width - r.height) < 0.3 * r.width && r.area() > 100 && r.area() < 10000) {
                R = Box(r);
                r_found = true;
                fixed_R_rect = r;
                fixed_R_center = R.center;
                is_R_fixed = true;
                return true;
            }
        }
        return false;
    }

    bool findFan(const Mat& bin) {
        Point2f center = fixed_R_center;

        Mat mask = Mat::zeros(bin.size(), CV_8UC1);
        circle(mask, center, cfg.broad_radius, Scalar(255), -1);
        circle(mask, center, cfg.broad_inner, Scalar(0), -1);
        Mat roi;
        bin.copyTo(roi, mask);

        vector<vector<Point>> contours;
        findContours(roi, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
        if (contours.empty()) { cout << "[调试] 状态二：无轮廓" << endl; return false; }

        int best_idx = -1;
        double max_score = 0;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area<cfg.min_fan_area || area>cfg.max_fan_area) continue;

            bool dist_ok = true;
            float avg_dist = 0;
            for (auto& p : contours[i]) {
                float d = norm(Point2f(p) - center);
                if (d<cfg.min_fan_dist || d>cfg.max_fan_dist) { dist_ok = false; break; }
                avg_dist += d;
            }
            if (!dist_ok) continue;
            avg_dist /= contours[i].size();

            vector<Point> approx;
            approxPolyDP(contours[i], approx, arcLength(contours[i], true) * cfg.approx_epsilon, true);
            double len = arcLength(approx, true);
            double circularity = 4 * CV_PI * area / (len * len);
            if (circularity < 0.6) continue;

            double score = area * circularity *
                (1 - fabs(avg_dist - (cfg.min_fan_dist + cfg.max_fan_dist) / 2)
                    / (cfg.max_fan_dist - cfg.min_fan_dist));

            if (score > max_score) {
                max_score = score;
                best_idx = i;
            }
        }
        if (best_idx == -1) { cout << "[调试] 状态二：无合格扇叶" << endl; return false; }

        auto& best = contours[best_idx];
        // 生成闭合轮廓并保存
        approxPolyDP(best, closed_fan_contour, arcLength(best, true) * cfg.approx_epsilon, true);
        // 保存拟合圆参数
        minEnclosingCircle(closed_fan_contour, circle_center, circle_radius);

        Point2f c_center;
        float c_radius;
        minEnclosingCircle(closed_fan_contour, c_center, c_radius);

        int rx = cvRound(c_center.x - c_radius);
        int ry = cvRound(c_center.y - c_radius);
        int rsize = cvRound(2 * c_radius);
        Rect square(rx, ry, rsize, rsize);

        fan_w = square.width;
        fan_h = square.height;

        float dmax = 0, dmin = 1e5;
        for (auto& p : best) {
            float d = norm(Point2f(p) - center);
            dmax = max(dmax, d);
            dmin = min(dmin, d);
        }
        outer_r = dmax;
        inner_r = dmin;
        avg_r = (dmax + dmin) / 2;
        radius_locked = true;

        cout << "[调试] 状态二：扇叶锁定" << endl;
        return true;
    }

    void updateFans() {
        angle += 0.03;
        for (int i = 0; i < 5; i++) {
            float t = angle + i * 72 * CV_PI / 180;
            Point2f dir(cos(t), sin(t));
            Point2f p = fixed_R_center + dir * avg_r;
            fans[i] = Box(Rect(p.x - fan_w / 2, p.y - fan_h / 2, fan_w, fan_h));
            fans[i].id = i;
        }
    }

    void updateR(const Mat& dil) {
        if (!is_R_fixed) return;
        R.rect = fixed_R_rect;
        R.center = fixed_R_center;
    }

    Mat getFanMask(const Size& sz) {
        Mat m = Mat::zeros(sz, CV_8UC1);
        if (!radius_locked) return m;
        circle(m, fixed_R_center, outer_r, Scalar(255), -1);
        circle(m, fixed_R_center, inner_r, Scalar(0), -1);
        return m;
    }

    void updateFanState(const Mat& bin) {
        Mat mask = getFanMask(bin.size());
        Mat roi;
        bin.copyTo(roi, mask);

        for (int i = 0; i < 5; i++) fan_state[i] = UNDEF;
        for (int i = 0; i < 5; i++) {
            Rect r = fans[i].rect;
            r.x -= cfg.roi_expand;
            r.y -= cfg.roi_expand;
            r.width += 2 * cfg.roi_expand;
            r.height += 2 * cfg.roi_expand;
            if (r.x < 0)r.x = 0; if (r.y < 0)r.y = 0;
            if (r.br().x > bin.cols) r.width = bin.cols - r.x;
            if (r.br().y > bin.rows) r.height = bin.rows - r.y;
            if (r.area() < 100) continue;

            Mat sub = roi(r);
            vector<vector<Point>> ct;
            findContours(sub, ct, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (ct.empty()) continue;
            double ma = 0;
            for (auto& c : ct) ma = max(ma, contourArea(c));
            if (ma > 100) fan_state[i] = TARGET;
        }
    }

    void process(Mat& frame, Mat& hsv_out, Mat& mask_out) {
        Mat hsv, bin, dil;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat m1, m2;
        inRange(hsv, Scalar(0, 173, 90), Scalar(32, 255, 255), m1);
        inRange(hsv, Scalar(170, 173, 90), Scalar(180, 255, 255), m2);
        bin = m1 | m2;
        Mat se = getStructuringElement(MORPH_RECT, Size(cfg.dilate_kernel, cfg.dilate_kernel));
        dilate(bin, dil, se);
        cvtColor(bin, hsv_out, COLOR_GRAY2BGR);

        if (phase == FIND_R) {
            cout << "[调试] 状态1：找R" << endl;
            putText(frame, "FIND R", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
            if (findR(dil)) {
                phase = FIND_FAN;
                cout << "[调试] R已永久固定" << endl;
                state2_start_time = steady_clock::now();
                state2_timer_started = true;
            }
        }
        else if (phase == FIND_FAN) {
            cout << "[调试] 状态2：找扇叶" << endl;
            putText(frame, "FIND FAN", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 1);
            updateR(dil);

            auto now = steady_clock::now();
            auto duration = duration_cast<milliseconds>(now - state2_start_time).count();
            putText(frame, "FAN STABILIZING: " + to_string(duration) + "ms",
                Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 1);

            bool fan_found = findFan(bin);
            if (fan_found && duration >= cfg.state2_min_duration) {
                phase = LOCK_RADIUS;
                cout << "[调试] 状态2→3：扇叶锁定且停留满" << duration << "ms" << endl;
            }
            else if (fan_found && duration < cfg.state2_min_duration) {
                cout << "[调试] 状态2：扇叶已找到，等待停留满" << cfg.state2_min_duration << "ms" << endl;
            }
        }
        else if (phase == LOCK_RADIUS) {
            cout << "[调试] 状态3：锁半径" << endl;
            putText(frame, "LOCK RADIUS", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 1);
            updateFans();
            phase = TRACK;
            cout << "[调试] 状态3→4：进入跟踪" << endl;
        }
        else if (phase == TRACK) {
            cout << "[调试] 状态4：跟踪（R固定）" << endl;
            putText(frame, "TRACKING", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1);
            updateR(dil);
            updateFans();
            updateFanState(bin);
        }

        mask_out = getFanMask(frame.size());
        cvtColor(mask_out, mask_out, COLOR_GRAY2BGR);
        draw(frame, hsv_out);
    }

    void draw(Mat& frame, Mat& hsv) {
        // 状态二：绘制闭合轮廓 + 拟合圆
        if (phase == FIND_FAN && !closed_fan_contour.empty()) {
            // 绘制闭合扇叶轮廓（红色，线宽2，完全闭合）
            drawContours(frame, vector<vector<Point>>{closed_fan_contour}, -1, Scalar(0, 0, 255), 2);
            // 绘制轮廓拟合圆（蓝色，线宽2）
            circle(frame, circle_center, cvRound(circle_radius), Scalar(255, 0, 0), 2);
        }

        rectangle(frame, fixed_R_rect, Scalar(0, 255, 0), 2);
        circle(frame, fixed_R_center, 4, Scalar(0, 255, 0), -1);
        rectangle(hsv, fixed_R_rect, Scalar(0, 255, 0), 2);

        if (radius_locked) {
            circle(frame, fixed_R_center, cvRound(outer_r), Scalar(0, 255, 255), 2);
            circle(frame, fixed_R_center, cvRound(inner_r), Scalar(0, 255, 255), 2);
            circle(frame, fixed_R_center, cvRound(avg_r), Scalar(255, 255, 0), 1);
        }

        for (int i = 0; i < 5; i++) {
            Scalar col(100, 100, 100);
            if (fan_state[i] == TARGET) col = Scalar(0, 0, 255);
            else if (fan_state[i] == HIT) col = Scalar(255, 0, 0);
            rectangle(frame, fans[i].rect, col, 1);
            if (fan_state[i] != UNDEF)
                circle(frame, fans[i].center, 5, Scalar(0, 255, 255), -1);
        }
    }
};

int main() {
    VideoCapture cap("Resources/video/Video Project 8.mp4");
    if (!cap.isOpened()) { cerr << "打开失败" << endl; return-1; }
    namedWindow("Energy Tracker", WINDOW_NORMAL);
    namedWindow("HSV View", WINDOW_NORMAL);
    EnergyTracker tracker;
    Mat frame, hsv, mask;
    while (cap.read(frame)) {
        tracker.process(frame, hsv, mask);
        imshow("Energy Tracker", frame);
        imshow("HSV View", hsv);
        if (waitKey(1) == 27)break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}