#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // 配置标定参数
    Size boardSize(7, 10);          // 棋盘格内角点数量 (列数, 行数)
    float squareSize = 19.5f;       // 每个棋盘格的物理尺寸
    int imagesToCollect = 10;        //需要采集的标定图像数量
    string cameraParamFile = "calibration.xml"; // 标定结果保存文件名

    // 初始化变量 
    vector<vector<Point3f>> objectPoints; // 实际坐标系下的三维点
    vector<vector<Point2f>> imagePoints;  // 图像坐标系下的二维点
    Size imageSize;                        // 图像尺寸

    // 生成标定板的世界坐标
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            obj.push_back(Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }

 //  打开摄像头
    VideoCapture cap(0); // 0为默认摄像头ID，若有多个摄像头请修改
    if (!cap.isOpened()) {
        cerr << "错误：无法打开摄像头！" << endl;
        return -1;
    }

    Mat frame, gray;
    int collected = 0;
    cout << "=== 相机标定程序启动 ===" << endl;
    cout << "请移动标定板，按 [空格键] 保存当前图像，按 [ESC] 退出采集" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 转换为灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        imageSize = gray.size();

        // 检测棋盘格角点
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        // 如果找到角点，进行亚像素优化
        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            // 在图像上绘制角点
            drawChessboardCorners(frame, boardSize, corners, found);
        }

        // 显示当前帧
        putText(frame, "Collected: " + to_string(collected) + "/" + to_string(imagesToCollect),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        imshow("Camera Calibration", frame);

        // 按键
        int key = waitKey(1);
        if (key == 27) break; // ESC退出
        if (key == 32 && found) { // 空格键保存图像
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
            collected++;
            cout << "已保存第 " << collected << " 张标定图像" << endl;
            if (collected >= imagesToCollect) {
                cout << "图像采集完成，开始标定..." << endl;
                break;
            }
        }
    }
    cap.release();
    destroyAllWindows();

    if (collected < 3) {
        cerr << "错误：采集的图像数量不足，无法标定！" << endl;
        return -1;
    }

    //  4. 执行相机标定
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F); // 内参矩阵
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F); // 畸变系数 (k1, k2, p1, p2, k3)
    vector<Mat> rvecs, tvecs; // 外参：旋转向量、平移向量

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs,
        CALIB_FIX_K4 | CALIB_FIX_K5); // 固定k4,k5为0（大多数镜头只需5个畸变系数）

    // 输出并保存标定结果
    cout << "\n=== 标定完成 ===" << endl;
    cout << "重投影误差 (RMS): " << rms << " (通常应 < 0.5 像素)" << endl;
    cout << "内参矩阵 (Camera Matrix):" << endl << cameraMatrix << endl;
    cout << "畸变系数 (Distortion Coefficients):" << endl << distCoeffs << endl;

    // 保存到XML文件
    FileStorage fs(cameraParamFile, FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "image_size" << imageSize;
    fs << "rms" << rms;
    fs.release();
    cout << "标定结果已保存至: " << cameraParamFile << endl;

    //  展示畸变校正效果
    cout << "\n按 [ESC] 退出校正预览" << endl;
    cap.open(0);
    Mat undistorted;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 畸变校正
        undistort(frame, undistorted, cameraMatrix, distCoeffs);

        // 拼接原图和校正图进行对比
        Mat compare;
        hconcat(frame, undistorted, compare);
        //putText(compare, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(compare, "Undistorted", Point(frame.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Distortion Correction", compare);
        if (waitKey(1) == 27) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}