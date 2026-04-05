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

// ========== 优化1：滑动条回调函数中设置参数（仅在拖动时调用，不影响循环） ==========
void onBrightnessTrackbar(int, void*) {
    // 亮度通过软件调节（convertTo），无需操作摄像头硬件
}

void onExposureTrackbar(int, void*) {
    // 曝光通过硬件调节（仅在拖动滑动条时调用，不拖慢循环）
    double exposureVal = exposure / 100.0; // 0-1映射
    // 注意：不同摄像头曝光范围不同，这里用相对值，部分摄像头可能需要调整
    cap.set(CAP_PROP_EXPOSURE, exposureVal * 10 - 5); // 示例：映射到-5到5的范围（可根据摄像头调整）
}

int main() {
    // ========== 优化2：明确设置摄像头参数（强制高帧率模式） ==========
    cap.open(1); // 0为默认摄像头，1为外接摄像头
    if (!cap.isOpened()) {
        cerr << "错误：无法打开摄像头！" << endl;
        return -1;
    }

    // 强制设置分辨率（根据摄像头支持调整，常用640x480/1280x720）
    int targetWidth = 640;
    int targetHeight = 480;
    cap.set(CAP_PROP_FRAME_WIDTH, targetWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, targetHeight);

    // 强制设置帧率（优先30fps，若摄像头不支持则自动降为支持的最大值）
    double targetFPS = 30.0;
    cap.set(CAP_PROP_FPS, targetFPS);

    // 获取实际生效的参数
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = targetFPS; // 兼容获取失败的情况
    cout << "摄像头参数：" << frameWidth << "x" << frameHeight << " @ " << fps << "fps" << endl;

    // 2. 创建显示窗口和滑动条
    namedWindow("Camera Control", WINDOW_AUTOSIZE); // 优化：用AUTOSIZE减少窗口渲染开销

    // 亮度滑动条
    createTrackbar("Brightness", "Camera Control", &brightness, 100, onBrightnessTrackbar);
    // 曝光滑动条
    createTrackbar("Exposure", "Camera Control", &exposure, 100, onExposureTrackbar);

    // ========== 优化3：视频写入器用更高效的编码（若支持H264优先用） ==========
    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v'); // 默认MP4编码
    // 若OpenCV编译时支持H264，可用下面的代码（更高效，文件更小）
    // int fourcc = VideoWriter::fourcc('H', '2', '6', '4'); 
    writer = VideoWriter(savePath, fourcc, fps, Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        cerr << "警告：视频写入器初始化失败，录制功能可能无法使用！" << endl;
    }

    // 4. 实时处理视频流
    Mat frame;
    double prevTime = getTickCount() / getTickFrequency();
    cout << "\n=== 操作说明 ===" << endl;
    cout << "1. 按 'r' 键开始/停止录制" << endl;
    cout << "2. 拖动滑动条调节亮度/曝光" << endl;
    cout << "3. 按 'q' 键退出程序" << endl;

    while (true) {
        // 读取摄像头帧（最核心的耗时操作，需保证优先执行）
        cap.read(frame);
        if (frame.empty()) {
            cerr << "错误：无法读取摄像头帧！" << endl;
            break;
        }

        // ========== 优化4：移除循环里的cap.set，仅保留软件亮度调节 ==========
        // 软件亮度调节（convertTo效率较高，不影响帧率）
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