# SimpleCNN
project2

## 代码功能
* 本代码通过卷积计算实现了对128 * 128分辨率大小图像的人脸识别功能，能够较为准确的分辨人脸和背景。
* 能够判断图片大小是否正确，以防止图片大小错误引发程序崩溃。
* 卷积层支持pad为0和1两种情况，kernel size只支持大小为3的情况。
* 在初版代码实现卷积功能的基础上，我对其进行了优化，将卷积计算转化为矩阵乘法运算，并通过之前作业中矩阵乘法的相关算法提升其运算速度。
* 整个程序运行较为稳定，目前测试中暂无Bug生成。

## 代码展示与测试
* 核心的三个函数：

  ```c++
  float* mat_1d(Mat&);
  float* conv_relu(int, int, int, float*, conv_param&);
  float* maxpool(int, int, int, float*);
  ```

  其中mat_1d是将Mat矩阵转化为一维数组的形式，并将BGR顺序调整至RGB。具体代码如下：

  ```c++
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
  ```

  这里使用了分块优化的算法加快其运行速度。

* conv_relu是最核心的卷积函数，同时包含了Relu过程。函数会根据pad值为0或1来判断是否需要进行padding过程。这里的计算采用的是最为直接的方式，代码量较大，这里不做展示，具体可见源文件。

* max_pool是池化函数，具体代码如下：

  ```c++
  loat* maxpool(int cha, int row, int col, float* inp)
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
  ```

  代码较为简单。

* 最后的flatten过程在主函数中进行，最后可以得到背景和人脸的confidence scores。

  <img src="https://github.com/Silver439/SimpleCNN/blob/main/picture/Screenshot%202021-01-02%20111118.png" alt="Screenshot 2021-01-02 111118" style="zoom:50%;" />

<img src="https://github.com/Silver439/SimpleCNN/blob/main/picture/Screenshot%202021-01-02%20111145.png" alt="Screenshot 2021-01-02 111145" style="zoom:50%;" />



* 注：以上图片每张都对应两组测试结果，前一组是优化前，后一组是优化后。可以看到结果与图片特征基本相符，接下来我将简单介绍一下我的优化思路。

## 优化代码：

* 在卷积运算中，将kernel视为一个窗口，该窗口每移动一次就会得到9个数与其相乘，我将这九个数拉直为一行，例如矩阵大小为8 * 8，我就可以得到64行，从而生成一个64 * 9 的矩阵m_matrix。由于每次卷积都会有多个kernel，例如16个，我们将每个kernel的9个数拉直为一列，这样便得到了一个9 * 16的矩阵k_matrix。从而可以将卷积运算转化为矩阵乘法。将每一个通道视为一层平面，则每层平面的计算都可以通过这种方式进行，最后将所得结果相加再加上bias就可以得到最终结果。

* 注：为了提升访存的连续性，我将kernel也展开为行的形式，这样矩阵乘法由行乘列变为行乘行。然后再通过分块的算法继续提升其运算速率。

* 具体代码实现如下：

  两个核心函数：

  ```c++
  float* quick_conv_relu(int cha, int row, int col, float* inp, conv_param& cp);
  void matrixproduct(float* C, float* A,int r,int col,int c,float* B);
  ```

  quick_conv_relu是改进后的卷积运算函数，matrixproduct是矩阵乘法运算函数。quick_conv_relu中首先将原数据转化为m_matrix,再将kernel数据转化为k_matrix,代码如下：

  ```c++
  int n = 0;
  		m_matrix = new float[row / st * col / st * ks * ks]{ 0 };
  		for (int k = 1; k <= row; k += st)
  		{
  			for (int l = 1; l <= col; l += st)
  			{
  				int fn2 = (loop) * (row + 2) * (col + 2) + k * (col + 2) + l;
  				m_matrix[n] = pd_inp[fn2-col-3];
  				m_matrix[n+1] = pd_inp[fn2-col-2];
  				m_matrix[n+2] = pd_inp[fn2-col-1];
  				m_matrix[n+3] = pd_inp[fn2-1];
  				m_matrix[n+4] = pd_inp[fn2];
  				m_matrix[n+5] = pd_inp[fn2 + 1];
  				m_matrix[n+6] = pd_inp[fn2+col+1];
  				m_matrix[n+7] = pd_inp[fn2+col+2];
  				m_matrix[n+8] = pd_inp[fn2+col+3];
  				n += 9;
  		    }
  	    }
  		int n2 = 0;
  		k_matrix = new float[o_cha * ks * ks]{0};
  		for (int i = 0; i < o_cha; i++)
  		{
  			int fn = i * i_cha * ks * ks + loop * ks * ks;
  			for (int j = 0; j < ks*ks; j++)
  			{
  				k_matrix[n2++] = cp.p_weight[fn + j];
  			}
  		}
  ```

  这样得到两个矩阵，直接通过matrixproduct函数进行矩阵乘法运算，代码如下：

  ```c++
  void matrixproduct(float* C, float* A, int r, int col, int c, float* B)
  {
  	for (int i = 1; i <= c; i+=4)
  	{
  		for (int j = 1; j <= r; j+=4)
  		{
  			for (int k = 0; k < col; k++)
  			{
  				C[(i - 1) * r + j - 1] += A[9 * (j - 1) + k] * B[9 * (i - 1) + k];
  				C[(i - 1) * r + j] += A[9 * (j) + k] * B[9 * (i - 1) + k];
  				C[(i - 1) * r + j + 1] += A[9 * (j + 1) + k] * B[9 * (i - 1) + k];
  				C[(i - 1) * r + j + 2] += A[9 * (j + 2) + k] * B[9 * (i - 1) + k];
  				C[(i) * r + j - 1] += A[9 * (j - 1) + k] * B[9 * (i) + k];
  				C[(i) * r + j] += A[9 * (j)+k] * B[9 * (i) + k];
  				C[(i) * r + j + 1] += A[9 * (j + 1) + k] * B[9 * (i) + k];
  				C[(i) * r + j + 2] += A[9 * (j + 2) + k] * B[9 * (i) + k];
  				C[(i + 1) * r + j - 1] += A[9 * (j - 1) + k] * B[9 * (i + 1) + k];
  				C[(i + 1) * r + j] += A[9 * (j)+k] * B[9 * (i + 1) + k];
  				C[(i + 1) * r + j + 1] += A[9 * (j + 1) + k] * B[9 * (i + 1) + k];
  				C[(i + 1) * r + j + 2] += A[9 * (j + 2) + k] * B[9 * (i + 1) + k];
  				C[(i + 2) * r + j - 1] += A[9 * (j - 1) + k] * B[9 * (i + 2) + k];
  				C[(i + 2) * r + j] += A[9 * (j)+k] * B[9 * (i + 2) + k];
  				C[(i + 2) * r + j + 1] += A[9 * (j + 1) + k] * B[9 * (i + 2) + k];
  				C[(i + 2) * r + j + 2] += A[9 * (j + 2) + k] * B[9 * (i + 2) + k];
  			}
  		}
  	}
  }
  ```

  由此可得结果。通过第二部分的测试结果可以看出计算答案与优化前相同。由于数据量太小，加速效果显得不太明显。但考虑到优化代码相比原版做了更多转化，而最终总时间还要少于原版代码，可以推出在数据量较大的情况下优化代码将会比原版代码速度上有较大提升。（因为在数据量的大的情况下计算对于时间的影响更大，转化过程的时间复杂度较低，影响较小）我们可以使用尝试使用openblas加速矩阵乘法运算，所得结果如下：
  
  <img src="https://github.com/Silver439/SimpleCNN/blob/main/picture/Screenshot%202021-01-09%20103113.png" alt="Screenshot 2021-01-09 103113" style="zoom:60%;" />
  
  可以看到速度又有了进一步的提升，且结果均正确。由此看来这样的优化思路的确是可行的。

## 结语：

* 以上便是本程序的全部内容。非常感谢于老师这一学期的教学以及精心设计的项目作业。这一个学期的学习使我感到非常愉悦，并且大大扩宽了我的眼界，使我对于编程的理解不再简单停留在简单的通过写代码获取答案上，而是开始关注代码实现的原理和效率。由于我并非计系学生，今后应该没有机会再选这样的课程，但我相信在可以预见的未来中，本学期的c++学习所给我带来的能力与眼界方面的提升将会让我的后续学习受益匪浅。
* 再次感谢，手动比心。


