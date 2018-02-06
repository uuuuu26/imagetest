#include<opencv2\opencv.hpp>   
#include<opencv2\highgui\highgui.hpp>
#include <list>  
#include <cmath>  

using namespace std;
using namespace cv;



//边缘检测
int main()
{
	Mat img1_after, img2_after;
	Mat img = imread("79f475d41070489b25f8ae0316723c1e9e5adf1400010b02205cd09e3c3fcda9eafdab588242edad75f4e3079597d143.jpg");
	imshow("原始图", img);


	Mat img1(img.rows, img.cols, CV_8UC3);
	img(Rect(0, 0, 344, 344)).copyTo(img1);
	imshow("img1", img1);
	Mat img2_small(img.rows, img.cols, CV_8UC3);
	img(Rect(0, 344, 115, 40)).copyTo(img2_small);
	Mat img2_big;
	resize(img2_small, img2_big, Size(), 3, 3);
	imshow("img2", img2_big);

	Mat DstPic, edge, grayImage;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst, out, out_1, out_above, out_down;
	//梯度边缘算法
	Sobel(img1, grad_x, CV_16S, 1, 0, 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(img1, grad_y, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	//imshow("整体方向soble", dst);
	//转灰度
	DstPic.create(dst.size(), dst.type());
	cvtColor(dst, grayImage, COLOR_BGR2GRAY);
	//滤波
	bilateralFilter(grayImage, out, 25, 25 * 2, 25 / 2);
	//imshow("gauss", out);
	//锐化
	Mat kernel(3, 3, CV_32F, Scalar(-1));
	kernel.at<float>(1, 1) = 8.9;
	filter2D(out, out_1, out.depth(), kernel);
	//imshow("ruihua", out_1);
	//二值化
	int th = 70;
	threshold(out_1, img1_after, th, 255, CV_THRESH_BINARY_INV);
	imshow("tupian", img1_after);


	//转灰度
	DstPic.create(img2_big.size(), img2_big.type());
	cvtColor(img2_big, grayImage, COLOR_BGR2GRAY);
	//滤波
	bilateralFilter(grayImage, out_1, 25, 25 * 2, 25 / 2);
	//二值化
	th = 180;
	threshold(out_1, out, th, 255, CV_THRESH_BINARY_INV);

	//0bai255hei
	//文字处理，取出四个单独文字

	int rows, cols;
	rows = out.rows;
	cols = out.cols;
	int cnt_row, cnt_cols, i;
	uchar *p;
	int quali[345] = { 0 };
	int devide[345] = { 0 };
	list <int>::iterator cut_cnt;
	list<int>cut;
	char* name[7] = { "cut0","cut1", "cut2", "cut3", "cut4", "cut5", "cut6" };
	int cut_confirmed[10] = { 0 };
	for (cnt_row = 0; cnt_row < rows; cnt_row++) {
		p = out.ptr<uchar>(cnt_row);
		for (cnt_cols = 0; cnt_cols < cols; cnt_cols++) {
			p[cnt_cols] = 255 - p[cnt_cols];
			quali[cnt_cols] += p[cnt_cols];
		}
	}
	for (cnt_cols = 1; cnt_cols < cols - 1; cnt_cols++) {
		if (abs(quali[cnt_cols] - quali[cnt_cols - 1]) < 255 && abs(quali[cnt_cols ] - quali[cnt_cols+1]) < 255 && quali[cnt_cols] > 30000)
			devide[cnt_cols] = 1;
		else
			devide[cnt_cols] = 0;
	}
	devide[cols-1] = devide[0] = 1;
	//imshow("2", out);
	waitKey(50);
	for (cnt_cols = 0; cnt_cols < cols; cnt_cols++) {
		if (devide[cnt_cols] == 1)devide[cnt_cols] = 0;
		else break;
	}
	for (cnt_cols = cols-1; cnt_cols >0; cnt_cols--) {
		if (devide[cnt_cols] == 1)devide[cnt_cols] = 0;
		else break;
	}
	for (cnt_cols = 0; cnt_cols < cols - 1; cnt_cols++) {
		if (devide[cnt_cols] != devide[cnt_cols + 1]) {
			int j = cnt_cols;
			for (i = j + 1; i < cols - 1; i++) {
				cnt_cols++;
				if (devide[i] != devide[i + 1]) break;
			}
			cut.push_back((j + i) / 2 + 1);
		}
	}
	i = 0;
	for (cut_cnt = cut.begin(); cut_cnt != cut.end(); cut_cnt++) i++;

	//梯度边缘算法
	Sobel(out, grad_x, CV_16S, 1, 0, 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(out, grad_y, CV_16S, 0, 1, 1, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img2_after);
	//imshow("整体方向soble", dst);

	for (cnt_row = 0; cnt_row < rows; cnt_row++) {
		p = img2_after.ptr<uchar>(cnt_row);
		for (cnt_cols = 0; cnt_cols < cols; cnt_cols++) {
			p[cnt_cols] = 255 - p[cnt_cols];
		}
	}


	Mat img2_cut[7];
	//for(i=0;i<7;i++){img2_cut[i] = Mat(img2_after.rows, img2_after.cols, CV_8UC1);}
	
	cut.push_front(0);
	cut.push_back(344);
	i = -1;
	for (cut_cnt = cut.begin(); cut_cnt != cut.end(); cut_cnt++) {
		i++;
		cut_confirmed[i] = *cut_cnt;
	}	
	for (i = 1; cut_confirmed[i] != 0; i++) {
		img2_after(Rect(cut_confirmed[i - 1], 0, (cut_confirmed[i] - cut_confirmed[i - 1]), rows)).copyTo(img2_cut[i-1]);
		
	//	img2_after(Rect(cut_confirmed[i-1], 0, cut_confirmed[i]- cut_confirmed[i-1], cols)).copyTo(img2_cut[i]);
	}
	for (i = 1; cut_confirmed[i] != 0; i++)	imshow(name[i-1], img2_cut[i-1]);






	waitKey(0);
}
