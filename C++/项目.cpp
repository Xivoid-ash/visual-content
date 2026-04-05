#include <iostream>   
#include <cmath>      
#include <string>     
#include <iomanip>   

using namespace std;

// 点结构体：存储二维坐标(x,y)
class Point {
public:
    double P_x;  // 点的x坐标
    double P_y;  // 点的y坐标

    // 坐标初始化为(0,0)
    Point() : P_x(0), P_y(0) {}

    //初始化指定坐标
    Point(double x, double y) : P_x(x), P_y(y) {}
};

// 存储装甲板基础属性
struct Rect {
    int R_ID;          // 装甲板数字ID（1-6）
    int R_color;       // 装甲板颜色：0=蓝色，1=红色
    Point R_point;     // 装甲板左上角坐标点
    double R_width;    // 装甲板宽度
    double R_hight;    // 装甲板高度

    // 初始化矩形属性
    Rect(int ID, int color, const Point& p, double width, double hight)
        : R_ID(ID), R_color(color), R_point(p), R_width(width), R_hight(hight) {
    }
};

// 封装装甲板计算
class Armor {
private:
    Rect rect; 

public:
    // 通过Rect结构体初始化装甲板
    Armor(const Rect& r) : rect(r) {}

    //装甲板中心点坐标
    Point Armor_Central_Point() const {
        // 中心x = 左上角x + 宽度/2
        double center_x = rect.R_point.P_x + rect.R_width / 2.0;
        // 中心y = 左上角y + 高度/2
        double center_y = rect.R_point.P_y + rect.R_hight / 2.0;
        return Point(center_x, center_y);
    }

    // 装甲板对角线长度
    double Diagonal() const {
        double diagonal = sqrt(pow(rect.R_width, 2) + pow(rect.R_hight, 2));
        return round(diagonal * 100) / 100;
    }

    // 获取装甲板四个角坐标
    void Armor_Around_Point(Point points[4]) const {
        // 左上角
        points[0] = rect.R_point;
        // 右上角
        points[1] = Point(rect.R_point.P_x + rect.R_width, rect.R_point.P_y);
        // 右下角
        points[2] = Point(rect.R_point.P_x + rect.R_width, rect.R_point.P_y + rect.R_hight);
        // 左下角
        points[3] = Point(rect.R_point.P_x, rect.R_point.P_y + rect.R_hight);
    }

    // 装甲板颜色字符串
    string Armor_Color() const {
        // 0返回蓝色，1返回红色
        return (rect.R_color == 0) ? "蓝" : "红";
    }

    // 获取装甲板ID
    int Armor_ID() const {
        return rect.R_ID;
    }
};

int main() {
    int id, color;            // 装甲板ID、颜色
    double x, y, width, height;  // 左上角坐标、宽、高

    cin >> id >> color;
    cin >> x >> y >> width >> height;

    Rect rect(id, color, Point(x, y), width, height);
    Armor armor(rect);
    //输出
    cout << "ID：" << armor.Armor_ID() << " 颜色：" << armor.Armor_Color() << endl;
    Point center = armor.Armor_Central_Point();
    double diagonal = armor.Diagonal();
    cout << "(" << center.P_x << "," << center.P_y << ")";
    cout << " 长度：" << fixed << setprecision(2) << diagonal << endl;
    Point points[4];
    armor.Armor_Around_Point(points);
    for (int i = 0; i < 4; ++i) {
        cout << "(" << points[i].P_x << "," << points[i].P_y << ")";
        if (i != 3) cout << " ";
    }
    cout << endl;
    system("pause");
    return 0;
}