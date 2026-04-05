#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

// 全局变量
int brightness = 50;    // 亮度初始值（0-100）
int exposure = 50;      // 曝光初始值（0-100）
VideoCapture cap;       // 摄像头对象
VideoWriter writer;     // 视频写入对象
bool isRecording = false;// 录制状态标记
string savePath = "recorded_video.mp4"; // 保存路径

void onBrightnessTrackbar(int, void*) {
}

void onExposureTrackbar(int, void*) {
    double exposureVal = exposure / 100.0; // 0-1映射
    cap.set(CAP_PROP_EXPOSURE, exposureVal * 10 - 5);
}

int main() {
    cap.open(1); // 0为默认摄像头，1为外接摄像头
    if (!cap.isOpened()) {
        cerr << "错误：无法打开摄像头！" << endl;
        return -1;
    }

    // 设置分辨率
    int targetWidth = 640;
    int targetHeight = 480;
    cap.set(CAP_PROP_FRAME_WIDTH, targetWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, targetHeight);

    // 设置帧率
    double targetFPS = 30.0;
    cap.set(CAP_PROP_FPS, targetFPS);

    // 获取实际生效的参数
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = targetFPS; // 兼容获取失败的情况
    cout << "摄像头参数：" << frameWidth << "x" << frameHeight << " @ " << fps << "fps" << endl;

    // 2. 创建显示窗口和滑动条
    namedWindow("Camera Control", WINDOW_AUTOSIZE); // 用AUTOSIZE减少窗口渲染开销

    // 亮度滑动条
    createTrackbar("Brightness", "Camera Control", &brightness, 100, onBrightnessTrackbar);
    // 曝光滑动条
    createTrackbar("Exposure", "Camera Control", &exposure, 100, onExposureTrackbar);

    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v'); // MP4编码
    writer = VideoWriter(savePath, fourcc, fps, Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        cerr << "警告：视频写入器初始化失败，录制功能可能无法使用！" << endl;
    }

    // 处理视频流
    Mat frame;
    double prevTime = getTickCount() / getTickFrequency();
    cout << "\n=== 操作说明 ===" << endl;
    cout << "1. 按 'r' 键开始/停止录制" << endl;
    cout << "2. 拖动滑动条调节亮度/曝光" << endl;
    cout << "3. 按 'q' 键退出程序" << endl;

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            cerr << "错误：无法读取摄像头帧！" << endl;
            break;
        }

        // 软件亮度调节
        double brightVal = (brightness - 50) / 50.0;
        frame.convertTo(frame, -1, 1.0, brightVal * 255);

        // 计算并显示实时FPS
        double currTime = getTickCount() / getTickFrequency();
        double currFPS = 1.0 / (currTime - prevTime);
        prevTime = currTime;

        // 在画面上绘制信息
        string fpsText = "FPS: " + to_string(static_cast<int>(currFPS));
        string sizeText = "Size: " + to_string(frameWidth) + "x" + to_string(frameHeight);
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        putText(frame, sizeText, Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // 显示录制状态
        if (isRecording) {
            putText(frame, "RECORDING", Point(frameWidth - 150, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            if (writer.isOpened()) {
                writer.write(frame);
            }
        }

        // 显示处理后的帧
        imshow("Camera Control", frame);

        // 键盘事件处理
        char key = waitKey(1);
        if (key == 'q' || key == 27) { // 按q或ESC退出
            break;
        }
        else if (key == 'r') { // 按r切换录制状态
            isRecording = !isRecording;
            cout << (isRecording ? "开始录制" : "停止录制") << endl;
        }
    }

    // 5. 释放资源
    cap.release();
    writer.release();
    destroyAllWindows();
    cout << "程序已退出！" << endl;

    return 0;
}