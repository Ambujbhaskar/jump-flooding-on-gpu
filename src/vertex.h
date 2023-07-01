#include "fstream"
#include "cmath"
#include "iostream"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Vertex
{
public:
    int row, col;
    int r = 0, g = 0, b = 0;
    __host__ __device__ Vertex(){};
    __host__ __device__ Vertex(int r, int c)
    {
        row = r;
        col = c;
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

        return (sqrtf((row - _row) * (row - _row) + (col - _col) * (col - _col)));
    }
    __host__ __device__ int index(int width)
    {
        return row * width + col;
    }
    friend std::istream &operator>>(std::istream &is, Vertex &v)
    {
        is >> v.row >> v.col;
        return is;
    }
    friend std::ostream &operator<<(std::ostream &os, Vertex &v)
    {
        os << v.row << " " << v.col << "    " << v.r << " " << v.g << " " << v.b;
        return os;
    }
};