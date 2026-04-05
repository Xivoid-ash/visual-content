#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <unordered_map>

using namespace cv;
using namespace std;

// 单目标平滑滤波器
class CircleSmoothFilter
{
public:
    CircleSmoothFilter(int windowSize = 5) : m_windowSize(windowSize) {}

    void smooth(Point2f& inputCenter, float& inputRadius, bool detectOK)
    {
        if (!detectOK) {
            if (!m_centerHistory.empty()) {
                inputCenter = m_lastCenter;
                inputRadius = m_lastRadius;
            }
            return;
        }

        m_centerHistory.push_back(inputCenter);
        m_radiusHistory.push_back(inputRadius);
        if (m_centerHistory.size() > m_windowSize) {
            m_centerHistory.pop_front();
            m_radiusHistory.pop_front();
        }

        Point2f sumPt(0, 0);
        float sumR = 0;
        for (auto& p : m_centerHistory) sumPt += p;
        for (auto& r : m_radiusHistory) sumR += r;

        inputCenter = sumPt / (float)m_centerHistory.size();
        inputRadius = sumR / (float)m_radiusHistory.size();

        m_lastCenter = inputCenter;
        m_lastRadius = inputRadius;
    }

private:
    int m_windowSize;
    deque<Point2f> m_centerHistory;
    deque<float> m_radiusHistory;
    Point2f m_lastCenter;
    float m_lastRadius;
};

// 最小二乘法拟合圆（修复正确公式）
bool fitCircleLeastSquare(const vector<Point>& contour, Point2f& center, float& radius)
{
    int n = contour.size();
    if (n < 3) return false;

    double x1 = 0, y1 = 0, x2 = 0, y2 = 0, xy1 = 0;
    for (auto& p : contour) {
        double x = p.x, y = p.y;
        x1 += x; y1 += y; x2 += x * x; y2 += y * y; xy1 += x * y;
    }

    double C = n * x2 - x1 * x1;
    double D = n * xy1 - x1 * y1;
    double E = n * (x2 * x1 + y2 * x1) - x1 * (x2 + y2);
    double G = n * y2 - y1 * y1;
    double H = n * (x2 * y1 + y2 * y1) - y1 * (x2 + y2);

    double det = C * G - D * D;
    if (fabs(det) < 1e-8) return false;

    double a = (D * H - E * G) / det;
    double b = (D * E - C * H) / det;
    center.x = a;
    center.y = b;

    double r = 0;
    for (auto& p : contour) r += norm(p - center);
    radius = r / n;
    return radius > 5 && radius < 300;
}

// 预处理
Mat preProcessImage(const Mat& src)
{
    Mat hsv, mask;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    Scalar l1(0, 100, 100), u1(10, 255, 255);
    Scalar l2(160, 100, 100), u2(179, 255, 255);
    Mat m1, m2;
    inRange(hsv, l1, u1, m1);
    inRange(hsv, l2, u2, m2);
    mask = m1 | m2;

    medianBlur(mask, mask, 3);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

// 多目标检测
vector<tuple<Point2f, float, Rect>> detectMultiTargets(const Mat& src, const Mat& mask)
{
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<tuple<Point2f, float, Rect>> result;

    for (auto& cnt : contours) {
        double area = contourArea(cnt);
        double peri = arcLength(cnt, true);
        if (area < 300 || area > 50000) continue;

        double circularity = 4 * CV_PI * area / (peri * peri);
        if (circularity < 0.55) continue;

        Point2f c; float r;
        if (!fitCircleLeastSquare(cnt, c, r)) continue;

        Rect rect(c.x - r * 3, c.y - r * 3, r * 6, r * 6);
        result.emplace_back(c, r, rect);
    }
    return result;
}

// 主函数
int main()
{
    VideoCapture cap("Resources/test_video.mp4");
    if (!cap.isOpened()) {
        cout << "视频打开失败！" << endl;
        return -1;
    }

    VideoWriter out("Resources/result_multi.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'),
        cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    unordered_map<int, CircleSmoothFilter> filters;
    vector<tuple<Point2f, float, Rect>> lastTargets;
    Mat frame, mask;
    int frameIdx = 0;

    while (cap.read(frame)) {
        frameIdx++;
        mask = preProcessImage(frame);
        auto current = detectMultiTargets(frame, mask);

        vector<tuple<Point2f, float, Rect>> displayList;

        // 多目标平滑 + 锁定
        for (int i = 0; i < current.size(); i++) {
            auto [c, r, rect] = current[i];
            if (filters.find(i) == filters.end()) filters[i] = CircleSmoothFilter(5);
            filters[i].smooth(c, r, true);
            displayList.emplace_back(c, r, rect);
        }

        // 丢失目标 → 显示最后锁定位置
        if (current.empty() && !lastTargets.empty()) {
            for (int i = 0; i < lastTargets.size(); i++) {
                auto [c, r, rect] = lastTargets[i];
                if (filters.count(i)) filters[i].smooth(c, r, false);
                displayList.emplace_back(c, r, rect);
            }
        }

        // 绘制所有锁定目标
        for (int i = 0; i < displayList.size(); i++) {
            auto [c, r, rect] = displayList[i];
            rectangle(frame, rect, Scalar(0, 0, 255), 2);
            circle(frame, c, 4, Scalar(0, 255, 255), -1);
            circle(frame, c, r, Scalar(255, 0, 0), 2);
            putText(frame, format("LOCK %d", i + 1),
                Point(rect.x, rect.y - 8), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        }

        lastTargets = displayList;
        out.write(frame);
        imshow("Multi Target Locked", frame);
        imshow("Mask", mask);

        char k = waitKey(1);
        if (k == 27) break;
    }

    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}