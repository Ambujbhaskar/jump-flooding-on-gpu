#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <cuda_runtime.h>
#include <vector>
#include <future>
#include <cstdlib>
#include <chrono>
#include <ctime>

#include "stb_image.h"
#include "stb_image_write.h"
#include "vertex.h"
#include "linesegment.h"

using namespace std;

#define threads 8 // no of threads for multithreading

int N = 1024;							 // image width & height (in px)
int blockSize = 64;				 // block size for stratified sampling of input points
int pointW = 0;						 // radius of points in post processing
int lineW = 0;						 // line width in post processing
bool CPU = false;					 // executes CPU algorithm if true, else executes on GPU
bool input = false;				 // toggle custom input
bool showSteps = false;		 // show intermediate steps for every jump
bool showLines = false;		 // toggle for lines generation
vector<Vertex> vertices;	 // stores voronoi vertices
vector<LineSegment> lines; // stores voronoi lines

void handleInput(int argc, char *argv[]);
void generate_random_voronoi_points(int blockSize);
void inputShapes();
void initialize_vertex_colors_CPU(vector<Vertex> &vertices);
void initialize_line_colors_CPU(vector<LineSegment> &lines);
void generate_voronoi(unsigned char *image_CPU_host, unsigned char *image_GPU_host);
void launch_CPU_parallel(unsigned char *image);
void initialize_voronoi_CPU_for_Chunk(unsigned char *image, Vec2 *image_dist, int thread);
void generate_voronoi_CPU_for_Chunk(unsigned char *image, unsigned char *image_out, Vec2 *image_dist, Vec2 *image_dist_out, int thread, int jump);
void launch_GPU_kernel(unsigned char *image_GPU_host);
void displayTiming(auto endTime, auto startTime);
void saveImage(int width, int height, unsigned char *bitmap, string name);
void render_voronoi_vertices_CPU(vector<Vertex> vertices, unsigned char *image, int pixelWidth);
void render_voronoi_lines_CPU(vector<LineSegment> lines, unsigned char *image, int pixelWidth);

int main(int argc, char *argv[])
{
	handleInput(argc, argv);																			// handle command line inputs of the program
	unsigned char *image_CPU_host = new unsigned char[N * N * 3]; // create RGB image
	unsigned char *image_GPU_host = new unsigned char[N * N * 3]; // create RGB image
	generate_random_voronoi_points(blockSize);										// generate vonoi input points using stratified sampling
	inputShapes();																								// take input and fill the shape arrays
	initialize_vertex_colors_CPU(vertices);												// set colors for vertices
	initialize_line_colors_CPU(lines);														// set colors for lines
	generate_voronoi(image_CPU_host, image_GPU_host);							// launch and time CPU or GPU Voronoi generation algorithm
	delete[] image_CPU_host;																			// delete the allocated image memory after saving
	delete[] image_GPU_host;																			// delete the allocated image memory after saving
}

void handleInput(int argc, char *argv[])
{
	if (argc == 2 && (string(argv[1]).compare("input") == 0))
	{
		input = true;
	}
	else if (argc == 2 && (string(argv[1]).compare("input") != 0))
	{
		blockSize = stoi(argv[1]);
	}
	else if (argc == 3 && (string(argv[2]).compare("lines") != 0))
	{
		blockSize = stoi(argv[1]);
		N = stoi(argv[2]);
	}
	else if (argc == 3 && (string(argv[2]).compare("lines") == 0))
	{
		blockSize = stoi(argv[1]);
		showLines = true;
	}
	else if (argc == 4 && (string(argv[1]).compare("input") == 0) && (string(argv[2]).compare("steps") == 0))
	{
		input = true;
		showSteps = true;
		N = stoi(argv[3]);
	}
	else if (argc == 4 && (string(argv[3]).compare("lines") == 0))
	{
		blockSize = stoi(argv[1]);
		N = stoi(argv[2]);
		showLines = true;
	}
	else if (argc == 4 && (string(argv[1]).compare("CPU") == 0))
	{
		CPU = true;
		N = stoi(argv[3]);
		input = true;
	}
	else if (argc == 6 && (string(argv[1]).compare("input") == 0) && (string(argv[2]).compare("steps") == 0))
	{
		input = true;
		showSteps = true;
		N = stoi(argv[3]);
		pointW = stoi(argv[4]);
		lineW = stoi(argv[5]);
	}
	else if (argc == 6 && (string(argv[1]).compare("CPU") == 0))
	{
		CPU = true;
		N = stoi(argv[3]);
		input = true;
		pointW = stoi(argv[4]);
		lineW = stoi(argv[5]);
	}
	else
	{
		cout << "\nUsage: \n"
				 << "    ./Voronoi input \n"
				 << "    ./Voronoi input steps [img_size]\n"
				 << "    ./Voronoi input steps [img_size] [point_size] [line_size]\n"
				 << "    ./Voronoi [block_size] \n"
				 << "    ./Voronoi [block_size] lines \n"
				 << "    ./Voronoi [block_size] [img_size] \n"
				 << "    ./Voronoi [block_size] [img_size] lines \n"
				 << "    ./Voronoi CPU input [img_size]\n"
				 << "    ./Voronoi CPU input [img_size] [point_size] [line_size]\n\n"
				 << "Note: After every run, seeds are saved in `../data/input.seeds`\n\n";
		exit(0);
	}

	cout << "Image: " << N << "x" << N << endl;
	if (!input)
	{
		cout << "blockSize: " << blockSize << "\n";
		if (showLines)
		{
			cout << "Generating Lines...\n";
		}
		else
		{
			cout << "Not generating Lines...\n";
		}
	}
}

void displayTiming(auto endTime, auto startTime)
{
	chrono::duration<double> elapsed_seconds = (endTime - startTime);
	cout << "Elapsed Time (sec): " << elapsed_seconds.count() << "s\n";
}

void generate_random_voronoi_points(int blockSize)
{
	if (input)
		return;
	srand(time(nullptr));
	int r1, r2, r3, r4;
	ofstream outputFileStream("../data/input.seeds");
	if (outputFileStream.good())
	{
		for (int i = 0; i < N; i += blockSize)
		{
			for (int j = 0; j < N; j += blockSize)
			{
				r1 = rand() % blockSize;
				r2 = rand() % blockSize;
				outputFileStream << "V " << (i + r1) << " " << (j + r2) << "\n";
			}
		}
		if (!showLines)
			return;
		for (int i = 0; i < N; i += blockSize)
		{
			for (int j = 0; j < N; j += blockSize * 2)
			{
				r1 = rand() % blockSize;
				r2 = rand() % blockSize;
				r3 = rand() % blockSize;
				r4 = rand() % blockSize;
				outputFileStream << "L " << (i + r1) << " " << (j + r2) << " " << (i + blockSize + r3) << " " << (j + blockSize + r4) << "\n";
			}
		}
	}
}

void inputShapes()
{
	ifstream sourceFileStream("../data/input.seeds"); // read input image
	if (sourceFileStream.good())
	{
		while (!sourceFileStream.eof())
		{
			string token;
			sourceFileStream >> token;
			if (token == "V")
			{
				vertices.emplace_back();
				sourceFileStream >> vertices.back();
			}
			else if (token == "L")
			{
				lines.emplace_back();
				sourceFileStream >> lines.back();
			}
		}
	}
}

void initialize_vertex_colors_CPU(vector<Vertex> &vertices)
{
	int red = 255, green = 140, blue = 210;
	const int vecSize = vertices.size();
	for (int i = 0; i < vecSize; i++)
	{
		vertices[i].setColor(
				(int)(red * ((float)(i + 125 * i + 1) / (float)vecSize)),
				(int)(green * ((float)(i - 100 * i + 1) / (float)vecSize)),
				(int)(blue * ((float)(i + 20 * i + 1) / (float)vecSize)));
	}
}

void initialize_line_colors_CPU(vector<LineSegment> &lines)
{
	int red = 213, green = 140, blue = 210;
	const int vecSize = lines.size();
	for (int i = 0; i < vecSize; i++)
	{
		lines[i].setColor(
				(int)(red * ((float)(i + 125 * i + 1) / (float)vecSize)),
				(int)(green * ((float)(i - 100 * i + 1) / (float)vecSize)),
				(int)(blue * ((float)(i + 20 * i + 1) / (float)vecSize)));
	}
}

void generate_voronoi(unsigned char *image_CPU_host, unsigned char *image_GPU_host)
{
	if (CPU)
	{
		auto startTime = chrono::steady_clock::now();				// start timing
		launch_CPU_parallel(image_CPU_host);								// create futures and wait for them to complete chunkwise computation
		auto endTime = chrono::steady_clock::now();					// end timing
		displayTiming(endTime, startTime);									// print timing results
		saveImage(N, N, image_CPU_host, "Voronoi_CPU.png"); // save the computed image
		if (pointW != 0 && lineW != 0)
		{
			render_voronoi_vertices_CPU(vertices, image_CPU_host, pointW); // render original vertices
			render_voronoi_lines_CPU(lines, image_CPU_host, lineW);				 // render original lines
			saveImage(N, N, image_CPU_host, "Voronoi_CPU_with_seeds.png"); // save the computed image
		}
	}
	else
	{
		launch_GPU_kernel(image_GPU_host); // launch GPU kernels
	}
}

void launch_CPU_parallel(unsigned char *image)
{
	unsigned char *image_out = new unsigned char[N * N * 3]; // create RGB image
	Vec2 *dist = new Vec2[N * N];														 // create distance vec2 image
	Vec2 *dist_out = new Vec2[N * N];												 // create distance vec2 image

	vector<future<void>> futureArray1;
	for (int thread = 0; thread < threads; thread++) // create futures which generate voronoi for chunks of the image
		futureArray1.push_back(async(launch::async, initialize_voronoi_CPU_for_Chunk, image, dist, thread));
	for (future<void> &f : futureArray1) // wait for the futures to finish
		f.get();

	for (int jump = N / 2; jump > 0; jump /= 2)
	{
		vector<future<void>> futureArray;
		for (int thread = 0; thread < threads; thread++) // create futures which generate voronoi for chunks of the image
			futureArray.push_back(async(launch::async, [thread, jump, &image, &image_out, &dist, &dist_out]()
																	{ generate_voronoi_CPU_for_Chunk(image, image_out, dist, dist_out, thread, jump); }));
		for (future<void> &f : futureArray) // wait for the futures to finish
			f.get();

		for (int i = 0; i < N * N; i++)
		{
			image[i * 3] = image_out[i * 3];
			image[i * 3 + 1] = image_out[i * 3 + 1];
			image[i * 3 + 2] = image_out[i * 3 + 2];
			dist[i] = dist_out[i];
		}
	}
}
void initialize_voronoi_CPU_for_Chunk(unsigned char *image, Vec2 *image_dist, int thread)
{
	int chunkSize = ((N * N) / threads) * 3;
	int start = thread * chunkSize;
	int end = thread * chunkSize + chunkSize;

	for (int idx = start; idx < end; idx += 3)
	{
		int ind = idx/3;
		for (int i = 0; i < vertices.size(); i++)
		{
			Vertex v = vertices[i];
			float dst = v.dist(ind, N);
			if (dst < .1) // if seed is present, set position and color to seed's
			{
				image[ind * 3] = v.r;
				image[ind * 3 + 1] = v.g;
				image[ind * 3 + 2] = v.b;
				image_dist[ind] = Vec2(ind / N, ind % N);
				break;
			}
		}
		for (int i = 0; i < lines.size(); i++)
		{
			LineSegment l = lines[i];
			float dst = l.dist(ind, N);
			if (dst < .5) // if line passes through the pixel, set position to here and color to line's
			{
				image[ind * 3] = l.r;
				image[ind * 3 + 1] = l.g;
				image[ind * 3 + 2] = l.b;
				image_dist[ind] = Vec2(ind / N, ind % N);
				break;
			}
		}
	}
}

void generate_voronoi_CPU_for_Chunk(unsigned char *image, unsigned char *image_out, Vec2 *image_dist, Vec2 *image_dist_out, int thread, int jump)
{
	int chunkSize = ((N * N) / threads) * 3;
	int start = thread * chunkSize;
	int end = thread * chunkSize + chunkSize;

	for (int idx = start; idx < end; idx += 3)
	{
		// for each pixel in chunk
		int ind = idx/3;

		Vec2 d = image_dist[ind];
		Vec2 p(ind / N, ind % N);
		unsigned char Rmin = 0, Gmin = 0, Bmin = 0;
		int distance[] = {-1, 0, 1};
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				int r = ind / N + (distance[i] * jump);
				int c = ind % N + (distance[j] * jump);
				if (
						r < N && c < N && r >= 0 && c >= 0 &&
						d.dist(image_dist[N * r + c], p) <= d.dist(d, p))
				{
					d = image_dist[N * r + c];
					Rmin = image[(N * r + c) * 3];
					Gmin = image[(N * r + c) * 3 + 1];
					Bmin = image[(N * r + c) * 3 + 2];
				}
			}
		}

		image_dist_out[ind] = d;			 // store closest seed's vec2
		image_out[ind * 3] = Rmin;		 // store its color Red
		image_out[ind * 3 + 1] = Gmin; // store its color Green
		image_out[ind * 3 + 2] = Bmin; // store its color Blue
	}
}

__global__ void generate_voronoi_GPU(unsigned char *image, unsigned char *image_out, Vec2 *image_dist, Vec2 *image_dist_out, int N, int jump)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	Vec2 d = image_dist[idx];
	Vec2 p(idx / N, idx % N);
	unsigned char Rmin = 0, Gmin = 0, Bmin = 0;
	int distance[] = {-1, 0, 1};
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int r = idx / N + (distance[i] * jump);
			int c = idx % N + (distance[j] * jump);
			if (
					r < N && c < N && r >= 0 && c >= 0 &&
					d.dist(image_dist[N * r + c], p) <= d.dist(d, p))
			{
				d = image_dist[N * r + c];
				Rmin = image[(N * r + c) * 3];
				Gmin = image[(N * r + c) * 3 + 1];
				Bmin = image[(N * r + c) * 3 + 2];
			}
		}
	}

	image_dist_out[idx] = d;			 // store closest seed's vec2
	image_out[idx * 3] = Rmin;		 // store its color Red
	image_out[idx * 3 + 1] = Gmin; // store its color Green
	image_out[idx * 3 + 2] = Bmin; // store its color Blue
}

__global__ void initialize_seeds_GPU(unsigned char *image, Vec2 *image_dist, Vertex *d_vertices, int VL, LineSegment *d_lines, int LL, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	image_dist[idx] = Vec2(4500, 4500); // initialize a far away seed position

	for (int i = 0; i < VL; i++)
	{
		Vertex v = d_vertices[i];
		float dst = v.dist(idx, N);
		if (dst < .1) // if seed is present, set position and color to seed's
		{
			image[idx * 3] = v.r;
			image[idx * 3 + 1] = v.g;
			image[idx * 3 + 2] = v.b;
			image_dist[idx] = Vec2(idx / N, idx % N);
			break;
		}
	}
	for (int i = 0; i < LL; i++)
	{
		LineSegment l = d_lines[i];
		float dst = l.dist(idx, N);
		if (dst < .5) // if line passes through the pixel, set position to here and color to line's
		{
			image[idx * 3] = l.r;
			image[idx * 3 + 1] = l.g;
			image[idx * 3 + 2] = l.b;
			image_dist[idx] = Vec2(idx / N, idx % N);
			break;
		}
	}
}

__global__ void render_seeds_GPU(unsigned char *image, Vertex *d_vertices, int VL, LineSegment *d_lines, int LL, int N, int pointW, int lineW)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < VL; i++)
	{
		Vertex v = d_vertices[i];
		float dst = v.dist(idx, N);
		if (dst < pointW)
		{
			image[idx * 3] = 255;
			image[idx * 3 + 1] = 255;
			image[idx * 3 + 2] = 255;
			break;
		}
	}
	for (int i = 0; i < LL; i++)
	{
		LineSegment l = d_lines[i];
		float dst = l.dist(idx, N);
		if (dst < lineW)
		{
			image[idx * 3] = 255;
			image[idx * 3 + 1] = 255;
			image[idx * 3 + 2] = 255;
			break;
		}
	}
}

void launch_GPU_kernel(unsigned char *image_GPU_host)
{
	Vertex *h_vertices = vertices.data();
	LineSegment *h_lines = lines.data();
	float *h_image_dist = new float[N * N];

	unsigned char *d_image;
	unsigned char *d_image_out;
	Vec2 *d_image_dist;
	Vec2 *d_image_dist_out;
	Vertex *d_vertices;
	LineSegment *d_lines;

	cudaMalloc((void **)&d_image, sizeof(unsigned char) * N * N * 3);
	cudaMalloc((void **)&d_image_out, sizeof(unsigned char) * N * N * 3);
	cudaMalloc((void **)&d_image_dist, sizeof(Vec2) * N * N);
	cudaMalloc((void **)&d_image_dist_out, sizeof(Vec2) * N * N);
	cudaMalloc((void **)&d_vertices, sizeof(Vertex) * vertices.size());
	cudaMalloc((void **)&d_lines, sizeof(LineSegment) * lines.size());

	cudaMemcpy(d_vertices, h_vertices, sizeof(Vertex) * vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lines, h_lines, sizeof(LineSegment) * lines.size(), cudaMemcpyHostToDevice);

	int blockDim = 1024;
	int gridDim = ceil(N * N / blockDim);
	initialize_seeds_GPU<<<gridDim, blockDim>>>(d_image, d_image_dist, d_vertices, vertices.size(), d_lines, lines.size(), N);
	cudaError_t e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		cout << cudaGetErrorString(e) << "\n";
	}
	cudaMemcpy(d_image_out, d_image, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_image_dist_out, d_image_dist, sizeof(Vec2) * N * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(image_GPU_host, d_image, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToHost);

	cudaEvent_t start_gpu, end_gpu;
	float msecs_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);
	cudaEventRecord(start_gpu, 0);

	if (showSteps)
		saveImage(N, N, image_GPU_host, "Voronoi_GPU_jumpNone.png"); // save the computed image

	for (int jump = N / 2; jump > 0; jump /= 2)
	{
		generate_voronoi_GPU<<<gridDim, blockDim>>>(d_image, d_image_out, d_image_dist, d_image_dist_out, N, jump);
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess)
		{
			cout << cudaGetErrorString(e) << "\n";
		}
		cudaMemcpy(d_image, d_image_out, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_image_dist, d_image_dist_out, sizeof(Vec2) * N * N, cudaMemcpyDeviceToDevice);
		if (showSteps)
		{
			cudaMemcpy(image_GPU_host, d_image_out, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToHost);
			string name = "Voronoi_GPU_jump" + to_string(jump) + ".png";
			saveImage(N, N, image_GPU_host, name); // save the computed image
		}
	}
	cudaMemcpy(image_GPU_host, d_image_out, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToHost);

	cudaEventRecord(end_gpu, 0); // timing end and display
	cudaEventSynchronize(end_gpu);
	cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(end_gpu);
	printf("Elapsed Time (secs): %fs.\n", msecs_gpu / 1000.0);

	saveImage(N, N, image_GPU_host, "Voronoi_GPU.png"); // save the computed image

	if (pointW != 0 && lineW != 0)
	{
		render_seeds_GPU<<<gridDim, blockDim>>>(d_image_out, d_vertices, vertices.size(), d_lines, lines.size(), N, pointW, lineW);
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess)
		{
			cout << cudaGetErrorString(e) << "\n";
		}
		cudaMemcpy(image_GPU_host, d_image_out, sizeof(unsigned char) * N * N * 3, cudaMemcpyDeviceToHost);
		saveImage(N, N, image_GPU_host, "Voronoi_GPU_with_seeds.png"); // save the computed image
	}
}

void render_voronoi_vertices_CPU(vector<Vertex> vertices, unsigned char *image, int pixelWidth)
{
	if (pixelWidth == 0)
		return;
	for (int i = 0; i < N * N * 3; i += 3)
	{
		// for each pixel
		for (auto v : vertices)
		{
			if (v.dist(i / 3, N) < pixelWidth)
			{
				image[i] = 255;
				image[i + 1] = 255;
				image[i + 2] = 255;
				break;
			}
		}
	}
}

void render_voronoi_lines_CPU(vector<LineSegment> lines, unsigned char *image, int pixelWidth)
{
	if (pixelWidth == 0)
		return;
	for (int i = 0; i < N * N * 3; i += 3)
	{
		// for each pixel
		for (auto l : lines)
		{
			if (l.dist(i / 3, N) < pixelWidth)
			{
				image[i] = 255;
				image[i + 1] = 255;
				image[i + 2] = 255;
				break;
			}
		}
	}
}

void saveImage(int width, int height, unsigned char *bitmap, string name)
{
	stbi_write_png(name.c_str(), width, height, 3, bitmap, width * 3);
	fprintf(stderr, "Image saved as: %s\n", name.c_str());
}
