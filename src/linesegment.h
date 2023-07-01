#include "fstream"
#include "cmath"
#include "iostream"
#include "vec2.h"

class LineSegment
{
public:
    int row1, col1;
    int row2, col2;
    int r = 0, g = 0, b = 0;
    __host__ __device__ LineSegment(){};
    __host__ __device__ LineSegment(int r1, int c1, int r2, int c2)
    {
        row1 = r1;
        col1 = c1;
        row2 = r2;
        col2 = c2;
    };
    __host__ __device__ void setColor(int R, int G, int B)
    {
        r = R;
        g = G;
        b = B;
    }
    __host__ __device__ float dist(int index, int width)
    {
        int _col = index % width;
        int _row = index / width;

        Vec2 p(_col, _row);
        Vec2 w(col1, row1);
        Vec2 v(col2, row2);
        // length of line segment = 0
        float l = (row1 - row2) * (row1 - row2) + (col1 - col2) * (col1 - col2);
        if (l == 0)
        {
            return p.dist(p, v);
        }
        float x = p.dot(p - v, w - v);
        const float t = std::fmaxf((float)0.0, std::fminf((float)1.1, x / l));
        Vec2 proj = v + (w - v) * t;
        return p.dist(p, proj);
    }
    friend std::istream &operator>>(std::istream &is, LineSegment &l)
    {
        is >> l.row1 >> l.col1 >> l.row2 >> l.col2;
        return is;
    }
    friend std::ostream &operator<<(std::ostream &os, LineSegment &l)
    {
        os << l.row1 << " " << l.col1 << "  " << l.row2 << " " << l.col2 << "    " << l.r << " " << l.g << " " << l.b;
        return os;
    }
};