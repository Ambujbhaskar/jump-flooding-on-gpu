#include "cmath"
#include "iostream"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Vec2
{
public:
    float x, y;
    __host__ __device__ Vec2()
    {
        x = 4500;
        y = 4500;
    }
    __host__ __device__ Vec2(float X, float Y)
    {
        x = X;
        y = Y;
    }
    __host__ __device__ Vec2 operator+(Vec2 const &v)
    {
        Vec2 temp(0, 0);
        temp.x = x + v.x;
        temp.y = y + v.y;
        return temp;
    }
    __host__ __device__ Vec2 operator-(Vec2 const &v)
    {
        Vec2 temp(0, 0);
        temp.x = x - v.x;
        temp.y = y - v.y;
        return temp;
    }
    __host__ __device__ Vec2 operator*(float const &a)
    {
        Vec2 temp(0, 0);
        temp.x = x * a;
        temp.y = y * a;
        return temp;
    }
    __host__ __device__ float dist(Vec2 a, Vec2 b)
    {
        return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }
    __host__ __device__ float dot(Vec2 a, Vec2 b)
    {
        return (a.x * b.x) + (a.y * b.y);
    }
};