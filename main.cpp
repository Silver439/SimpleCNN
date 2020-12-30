#include"head.h"
using namespace std;
using namespace cv;

extern conv_param conv_params[3];
extern fc_param fc_params[1];

float* mat_1d(Mat&);
float* conv_relu(int, int, int, float*, conv_param&);
float* maxpool(int, int, int, float*);

int main()
{
	Mat img = imread("2.jpg");
	float* res = mat_1d(img);

	float* p0 = conv_relu(3, 128, 128, res, conv_params[0]);
	float* p1 = maxpool(16, 64, 64, p0);

	float* p2 = conv_relu(16, 32, 32, p1, conv_params[1]);
	float* p3 = maxpool(32, 32, 32, p2);

	float* p4 = conv_relu(32, 16, 16, p3, conv_params[2]);

	float* p5 = new float[2]{ 0 };
	for (int i = 0; i < 2048; i++)
	{
		p5[0] += fc_params[0].p_weight[i] * p4[i];
		p5[1] += fc_params[0].p_weight[i + 2048] * p4[i];
	}
	p5[0] += fc_params[0].p_bias[0];
	p5[1] += fc_params[0].p_bias[1];
	float conf1, conf2;
	conf1 = exp(p5[0]) / (exp(p5[0]) + exp(p5[1]));
	conf2 = exp(p5[1]) / (exp(p5[0]) + exp(p5[1]));
	cout << "backgrond:" << conf1 << endl;
	cout << "face:" << conf2 << endl;
}

float* mat_1d(Mat& img)
{
	int row = img.rows;
	int col = img.cols;
	int n = row * col;
	float* res = new float[3 * n];
	for (int i = 0; i < row; i++)
	{
		uchar* p = img.ptr<uchar>(i);
		for (int j = 0; j < col - 1; j += 4)
		{
			res[i * col + j] = (float)p[3 * j + 2] / (float)255;
			res[i * col + j + 1] = (float)p[3 * j + 5] / (float)255;
			res[i * col + j + 2] = (float)p[3 * j + 8] / (float)255;
			res[i * col + j + 3] = (float)p[3 * j + 11] / (float)255;
			res[n + i * col + j] = (float)p[3 * j + 1] / (float)255;
			res[n + i * col + j + 1] = (float)p[3 * j + 4] / (float)255;
			res[n + i * col + j + 2] = (float)p[3 * j + 7] / (float)255;
			res[n + i * col + j + 3] = (float)p[3 * j + 10] / (float)255;
			res[2 * n + i * col + j] = (float)p[3 * j] / (float)255;
			res[2 * n + i * col + j + 1] = (float)p[3 * j + 3] / (float)255;
			res[2 * n + i * col + j + 2] = (float)p[3 * j + 6] / (float)255;
			res[2 * n + i * col + j + 3] = (float)p[3 * j + 9] / (float)255;
		}
	}
	return res;
}

float* conv_relu(int cha, int row, int col, float* inp, conv_param &cp)
{
	int num = 0;
	int o_cha = cp.out_channels;
	int i_cha = cp.in_channels;
	int ks = cp.kernel_size;
	int st = cp.stride;
	if (cp.pad == 1)
	{
		float* pd_inp = new float[cha * (row * col + 2 * col + 2 * row + 4)]{ 0 };
		for (int i = 1; i <= cha; i++)
		{
			for (int j = 1; j <= row; j++)
			{
				for (int k = 1; k <= col; k++)
				{
					pd_inp[(i - 1) * (row + 2) * (col + 2) + j * (col + 2) + k] = inp[num++];
				}
			}
		}
		delete[] inp;
		float* res = new float[o_cha * row * col / st / st]{ 0 };
		for (int i = 0; i < o_cha; i++)
		{
			int forword_num = i * ks * ks * i_cha;
			for (int j = 1; j <= cha; j++)
			{
				int fn = forword_num + (j - 1) * ks * ks;
				for (int k = 1; k <= row; k += cp.stride)
				{
					for (int l = 1; l <= col; l += cp.stride)
					{
						int fn2 = (j - 1) * (row + 2) * (col + 2) + k * (col + 2) + l;
						res[i * row * col / st / st + (k - 1) * col / st / st + (l - 1) / st] += 
							pd_inp[fn2] * cp.p_weight[fn + 4] +
							pd_inp[fn2 - 1] * cp.p_weight[fn + 3] +
							pd_inp[fn2 + 1] * cp.p_weight[fn + 5] +
							pd_inp[fn2 - col - 2] * cp.p_weight[fn + 1] +
							pd_inp[fn2 + col + 2] * cp.p_weight[fn + 7] +
							pd_inp[fn2 - 1 - col - 2] * cp.p_weight[fn] +
							pd_inp[fn2 - 1 + col + 2] * cp.p_weight[fn + 6] +
							pd_inp[fn2 + 1 - col - 2] * cp.p_weight[fn + 2] +
							pd_inp[fn2 + 1 + col + 2] * cp.p_weight[fn + 8];
					}
				}
			}
		}
		int total = o_cha * row * col / st / st;
		for (int i = 0; i < o_cha; i++)
		{
			for (int j = 0; j < row * col / st / st; j++)
			{
				res[i * row * col / st / st + j] += cp.p_bias[i];
			}
		}
		for (int i = 0; i < total; i++)
		{
			if (res[i] < 0) {
				res[i] = 0;
			}
		}
		return res;
	}
	else
	{
		float* res = new float[o_cha * (row-2) * (col-2) / st / st]{ 0 };
		for (int i = 0; i < o_cha; i++)
		{
			int forword_num = i * ks * ks * i_cha;
			for (int j = 1; j <= cha; j++)
			{
				int fn = forword_num + (j - 1) * ks * ks;
				for (int k = 1; k <= row-2; k += cp.stride)
				{
					for (int l = 1; l <= col-2; l += cp.stride)
					{
						int fn2 = (j-1) * (row) * (col) + k * (col) + l;
						res[i * (row-2) * (col-2) / st / st + (k - 1) * (col-2) / st / st + (l - 1) / st] +=
							inp[fn2] * cp.p_weight[fn + 4] +
							inp[fn2 - 1] * cp.p_weight[fn + 3] +
							inp[fn2 + 1] * cp.p_weight[fn + 5] +
							inp[fn2 - col - 2] * cp.p_weight[fn + 1] +
							inp[fn2 + col + 2] * cp.p_weight[fn + 7] +
							inp[fn2 - 1 - col - 2] * cp.p_weight[fn] +
							inp[fn2 - 1 + col + 2] * cp.p_weight[fn + 6] +
							inp[fn2 + 1 - col - 2] * cp.p_weight[fn + 2] +
							inp[fn2 + 1 + col + 2] * cp.p_weight[fn + 8];
					}
				}
			}
		}
		int total = o_cha * (row-2) * (col-2) / st / st;
		for (int i = 0; i < o_cha; i++)
		{
			for (int j = 0; j < (row-2) * (col-2) / st / st; j++)
			{
				res[i * (row-2) * (col-2) / st / st + j] += cp.p_bias[i];
			}
		}
		for (int i = 0; i < total; i++)
		{
			if (res[i] < 0) {
				res[i] = 0;
			}
		}
		delete[] inp;
		return res;
	}
}

float* maxpool(int cha, int row, int col, float* inp)
{
	float* res = new float[cha * row * col / 4];
	for (int i = 0; i < cha; i++)
	{
		for (int j = 1; j <= row; j += 2)
		{
			for (int k = 1; k <= col; k += 2)
			{
				float max = inp[i * row * col + (j - 1) * col + k - 1];
				if (inp[i * row * col + (j - 1) * col + k] > max) max = inp[i * row * col + (j - 1) * col + k];
				if (inp[i * row * col + j * col + k - 1] > max) max = inp[i * row * col + j * col + k - 1];
				if (inp[i * row * col + j * col + k] > max) max = inp[i * row * col + j * col + k];
				res[i * row * col / 4 + (j - 1) * col / 4 + (k - 1) / 2] = max;
			}
		}
	}
	delete[] inp;
	return res;
}
