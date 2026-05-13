// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>
#include <fstream>

using namespace std;
using namespace cv;

wchar_t* projectPath;

typedef struct {
	uchar* grayscale_values;
	uchar count_grayscale_values;
} grayscale_mapping;

typedef struct {
	Mat contour;
	int length;
} contour_info;

typedef struct {
	int c_min, c_max, r_min, r_max;
} rectangle_coord;

typedef struct {
	Mat label;
	int no_labels;
} labels;

typedef struct {
	int x; //rand
	int y; //coloana
} Point1;

const Point1 neighbours4[] = {
	{0, 1},  // Dreapta
	{0, -1}, // Stânga
	{1, 0},  // Jos
	{-1, 0}  // Sus
};

const Point1 neighbours8[] = {
	{0, 1},   // Dreapta (0)
	{1, 1},   // Dreapta-Jos (1)
	{1, 0},   // Jos (2)
	{1, -1},  // Stânga-Jos (3)
	{0, -1},  // Stânga (4)
	{-1, -1}, // Stânga-Sus (5)
	{-1, 0},  // Sus (6)
	{-1, 1}   // Dreapta-Sus (7)
};

typedef struct {
	std::vector<Point1> border;
	std::vector<int> dir_vector;
}contour;

typedef struct {
	Mat src;
	int TH_area;
	double phi_LOW;
	double phi_HIGH;
} CallbackData;

struct ChainResults {
	std::vector<int> chainCode;
	std::vector<int> derivative;
};

// Mapăm vecinii conform standardului Freeman (0=E, 1=NE, 2=N, ...)
const Point1 freeman_neighbors[] = {
	{0, 1},   // 0: Est
	{-1, 1},  // 1: Nord-Est
	{-1, 0},  // 2: Nord
	{-1, -1}, // 3: Nord-Vest
	{0, -1},  // 4: Vest
	{1, -1},  // 5: Sud-Vest
	{1, 0},   // 6: Sud
	{1, 1}    // 7: Sud-Est
};

Point1 findPo(Mat src) {

	Point1 p0;
	p0.x = -1;
	p0.y = -1;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				p0.x = i;
				p0.y = j;
				return p0;
			}
		}
	}

	return p0;
}
Mat draw_contour(contour cnt, Mat src) {
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < cnt.border.size(); i++) {
		Point1 p = cnt.border[i];
		circle(dst, Point(p.y, p.x), 3, Scalar(0, 0, 255), -1);
	}
	return dst;
}
void display_codes(ChainResults res) {
	printf("\nCodul Inlantuit (n=%zu):\n", res.chainCode.size());
	for (int val : res.chainCode) printf("%d", val);

	printf("\n\nDerivata Codului Inlantuit:\n");
	for (int val : res.derivative) printf("%d", val);
	printf("\n");
}
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	// Fixăm o lățime vizibilă constantă, de exemplu 512 pixeli, 
		// indiferent dacă avem 10 sau 256 de coșuri
	int display_width = 512;
	Mat imgHist(hist_height, display_width, CV_8UC3, Scalar(255, 255, 255));

	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	// Calculăm cât de lată trebuie să fie fiecare bară
	float bin_w = (float)display_width / hist_cols;

	for (int i = 0; i < hist_cols; i++) {
		// Calculăm dreptunghiul pentru fiecare coș
		Point p1(i * bin_w, baseline);
		Point p2((i + 1) * bin_w - 1, baseline - cvRound(hist[i] * scale));

		// Desenăm un dreptunghi plin în loc de o singură linie
		rectangle(imgHist, p1, p2, CV_RGB(255, 0, 255), FILLED);
	}

	imshow(name, imgHist);
}

void ex1_3_change_gray_levels_additive()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				dst.at<uchar>(i, j) = min(val + 20,255);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}
void ex1_4_change_gray_levels_multiplicative()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = min(val * 2, 255);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);

		imwrite("D:/FACULTATE/PI/opencv/OpenCVApplication-VS2022_OCV490_basic/Images/negative_ex4.bmp", dst); 
		waitKey();
	}
}
void ex1_5_four_coloured_square()
{
	while (1)
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = Mat (256, 256, CV_8UC3);

		int height = 256;
		int width = 256;
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow

		//colorare alb
		for (int i = 0; i < 128; i++)
		{
			for (int j = 0; j < 128; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				unsigned char b = 255;
				unsigned char g = 255;
				unsigned char r = 255;
				pixel[0] = b;
				pixel[1] = g;
				pixel[2] = r;
				src.at<Vec3b>(i, j) = pixel;
			}
		}

		//rosu
		for (int i = 0; i < 128; i++)
		{
			for (int j = 128; j < 256; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				unsigned char b = 0;
				unsigned char g = 0;
				unsigned char r = 255;
				pixel[0] = b;
				pixel[1] = g;
				pixel[2] = r;
				src.at<Vec3b>(i, j) = pixel;
			}
		}

		//verde
		for (int i = 128; i < 256; i++)
		{
			for (int j = 0; j < 128; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				unsigned char b = 0;
				unsigned char g = 255;
				unsigned char r = 0;
				pixel[0] = b;
				pixel[1] = g;
				pixel[2] = r;
				src.at<Vec3b>(i, j) = pixel;
			}
		}

		//galben
		for (int i = 128; i < 256; i++)
		{
			for (int j = 128; j < 256; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				unsigned char b = 0;
				unsigned char g = 255;
				unsigned char r = 255;
				pixel[0] = b;
				pixel[1] = g;
				pixel[2] = r;
				src.at<Vec3b>(i, j) = pixel;
			}
		}



		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);

		waitKey();
	}
}
void ex1_6_inverse_of_matrix()
{
	int ok = 0;
	while (1)
	{
		
		double t = (double)getTickCount(); // Get the current time [s]

		float vals[9] = { 1.09f, 2.0f, 3.33f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
		Mat M(3, 3, CV_32FC1, vals);
		Mat M2;

		int height = M.rows;
		int width = M.cols;

		if (ok == 0)
		{
			invert(M, M2);
			std::cout << M2 << std::endl;

			//cand afisam o matrice cu valori float, acestea sunt normalizate in intervalul [0,1] pentru a putea fi vizualizate
			Mat visualM2;
			normalize(M2, visualM2, 0, 1, NORM_MINMAX);

			//redimensionam matricea pentru a putea fi vizualizata
			Mat resizedM2;
			resize(visualM2, resizedM2, Size(300, 300), 0, 0, INTER_NEAREST);
			imshow("M2", resizedM2);

			ok = 1;
		}

		//cand afisam o matrice cu valori float, acestea sunt normalizate in intervalul [0,1] pentru a putea fi vizualizate
		Mat visualM;
		normalize(M, visualM, 0, 1, NORM_MINMAX);

		//redimensionam matricea pentru a putea fi vizualizata
		Mat resizedM;
		resize(visualM, resizedM, Size(300, 300), 0, 0, INTER_NEAREST);
		imshow("M", resizedM);

		waitKey();

	}
}
void ex2_1_extract_RGB_channels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat dst_blue = Mat(height, width, CV_8UC1);
		Mat dst_green = Mat(height, width, CV_8UC1);
		Mat dst_red = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel_src = src.at<Vec3b>(i, j);
				uchar B = pixel_src[0];
				uchar G = pixel_src[1];
				uchar R = pixel_src[2];

				dst_blue.at<uchar>(i, j) = B;
				dst_green.at<uchar>(i, j) = G;
				dst_red.at<uchar>(i, j) = R;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("blue", dst_blue);
		imshow("green", dst_green);
		imshow("red", dst_red);

		waitKey();
	}
}
void ex2_3_grayscale_to_BW() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		int prag;

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		::scanf("%d", &prag);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel >= prag) 
					dst.at<uchar>(i, j) = 255;
				else dst.at<uchar>(i, j) = 0;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("black-white", dst);

		waitKey();
	}
}
void isInside(int x, int y) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		if (x >= 0 && x < width) {
			if (y >= 0 && y < height) {
				printf("Este in imagine");
			}
			else printf("Nu este in imagine");
		} else printf("Nu este in imagine");

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		printf("Dimensiunea imaginii este: %d, %d", width, height);

		imshow("input image", src);

		waitKey();
	}
}
void ex2_4_RBG_to_HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;
		Mat dst_H = Mat(height, width, CV_8UC1);
		Mat dst_S = Mat(height, width, CV_8UC1);
		Mat dst_V = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel_src = src.at<Vec3b>(i, j);
				uchar B = pixel_src[0];
				uchar G = pixel_src[1];
				uchar R = pixel_src[2];

				float B_norm = (float)B / 255;
				float G_norm = (float)G / 255;
				float R_norm = (float)R / 255;

				float M1 = max(R_norm, G_norm);
				float M = max(M1, B_norm);

				float m1 = min(R_norm, G_norm);
				float m = min(m1, B_norm);

				float C = M - m;
				float V = M;
				float S;
				float H;
				
				if (V != 0) {
					S = C / V;
				}
				else S = 0;

				if(C != 0) {
					if (M == R_norm) H = 60 * (G_norm - B_norm) / C;
					if (M == G_norm) H = 120 + 60 * (B_norm - R_norm) / C;
					if (M == B_norm) H = 240 + 60 * (R_norm - G_norm) / C;
				}
				else  // grayscale 
					H = 0;
				if(H < 0)
					H = H + 360;

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

				dst_H.at<uchar>(i, j) = saturate_cast<uchar>(H * 255.0f / 360.0f);
				dst_S.at<uchar>(i, j) = saturate_cast<uchar>(S * 255.0f);
				dst_V.at<uchar>(i, j) = saturate_cast<uchar>(V * 255.0f);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		printf("Dimensiunea imaginii este: %d, %d", width, height);

		imshow("input image", src);
		imshow("h", dst_H);
		imshow("s", dst_S);
		imshow("v", dst_V);

		waitKey();
	}
}
int* ex3_1_compute_hist(Mat img)
{
	//creem un vector de 256 de pozitii (cate una pentru fiecare valoare posibila a pixelilor) 
	// in care vom tine numarul de aparitii a fiecarei valori
    int* hist = (int*)calloc(256, sizeof(int));
    if (!hist) return nullptr; 

    int height = img.rows;
    int width = img.cols;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            uchar pixel = img.at<uchar>(i, j);
            hist[pixel]++;
        }
    }
    return hist;
}
float* ex3_2_compute_pdf(int* hist, int totalPixels)
{
	float* pdf = (float*)calloc(256, sizeof(float));
	if (!pdf) return nullptr;

	for (int i = 0; i < 256; i++)
	{
		pdf[i] = (float)hist[i] / totalPixels;
	}
	return pdf;
}
int* ex3_4_compute_hist_custom(Mat img, int no_bins)
{
	int* hist = (int*)calloc(no_bins, sizeof(int));
	if (!hist) return nullptr;
	int height = img.rows;
	int width = img.cols;

	float bin_size = 256.0f / no_bins;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar pixel = img.at<uchar>(i, j);
			int bin_idx = (int)(pixel / bin_size);

			if (bin_idx >= no_bins) 
				bin_idx = no_bins - 1;

			hist[bin_idx]++;
		}
	}
	return hist;
}
grayscale_mapping ex3_5_1_multi_level_thresholding(int wh, float th, float* pdf)
{
	uchar* temp_maxima = (uchar*)calloc(256, sizeof(uchar));
	uchar count = 0;
	temp_maxima[count++] = 0;

	for (int k = wh; k <= 255 - wh; k++) {
		float sum = 0;
		float max_in_window = -1.0f;

		for (int i = k - wh; i <= k + wh; i++) {
			sum += pdf[i]; // calculăm suma valorilor din fereastră pentru a putea calcula media
			if (pdf[i] > max_in_window) {
				max_in_window = pdf[i];
			}
		}

		float v = sum / (2 * wh + 1);
		if (pdf[k] > v + th && pdf[k] >= max_in_window) {
			if (k > 0) {
				temp_maxima[count++] = k;
			}
		}
	}

	if (temp_maxima[count - 1] != 255) {
		temp_maxima[count++] = 255;
	}

	grayscale_mapping mapping;
	mapping.count_grayscale_values = (uchar)count;
	mapping.grayscale_values = (uchar*)malloc(count * sizeof(uchar));

	for (int i = 0; i < count; i++) {
		mapping.grayscale_values[i] = (uchar)temp_maxima[i];
	}

	return mapping;
}
uchar ex3_5_2_find_closest_histogram_maximum(uchar old_pixel, grayscale_mapping map) {
	uchar* values = map.grayscale_values;
	int n = map.count_grayscale_values;

	uchar best_value = values[0];
	int best_dist = abs((int)old_pixel - (int)values[0]);

	for (int i = 1; i < n; i++)
	{
		int dist = abs((int)old_pixel - (int)values[i]);

		if (dist < best_dist) {
			best_dist = dist;
			best_value = values[i];
		}
		else break;
	}

	return best_value;
}
Mat ex3_5_3_draw_multiple_thresholding(Mat source, grayscale_mapping map) {
	Mat result = source.clone();

	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			uchar old_value = source.at<uchar>(i, j);
			result.at<uchar>(i, j) = ex3_5_2_find_closest_histogram_maximum(old_value, map);
		}
	}

	return result;
}
void ex3_6_1_diffuse_error(Mat& img, int r, int c, float error) {
	int rows = img.rows;
	int cols = img.cols;

	// Dreapta (7/16)
	if (c + 1 < cols)
		img.at<float>(r, c + 1) += error * 7 / 16.0f;

	if (r + 1 < rows) {
		// Jos-Stânga (3/16)
		if (c - 1 >= 0)
			img.at<float>(r + 1, c - 1) += error * 3 / 16.0f;

		// Jos (5/16)
		img.at<float>(r + 1, c) += error * 5 / 16.0f;

		// Jos-Dreapta (1/16)
		if (c + 1 < cols)
			img.at<float>(r + 1, c + 1) += error * 1 / 16.0f;
	}
}
float ex3_6_2_process_pixel_floyd(Mat& img, int r, int c, grayscale_mapping map) {
	float old_val = img.at<float>(r, c);

	// Limităm valoarea la [0, 255] pentru a putea căuta în mapping
	uchar clamped = (uchar)((std::max))(0.0f, ((std::min))(255.0f, old_val));

	// Folosim funcția ta existentă pentru a găsi cea mai apropiată nuanță dominantă
	uchar new_val = ex3_5_2_find_closest_histogram_maximum(clamped, map);

	// Actualizăm pixelul în matricea de lucru
	img.at<float>(r, c) = (float)new_val;

	// Returnăm eroarea (diferența dintre ce aveam și ce am pus)
	return old_val - (float)new_val;
}
Mat ex3_6_3_draw_floyd_steinberg(Mat source, grayscale_mapping map) {
	// Convertim sursa la float pentru a nu pierde precizia erorilor (care pot fi negative)
	Mat workImg;
	source.convertTo(workImg, CV_32FC1);

	for (int i = 0; i < workImg.rows; i++) {
		for (int j = 0; j < workImg.cols; j++) {
			// Pasul 1: Procesăm pixelul și obținem eroarea
			float error = ex3_6_2_process_pixel_floyd(workImg, i, j, map);

			// Pasul 2: Distribuim eroarea vecinilor
			ex3_6_1_diffuse_error(workImg, i, j, error);
		}
	}

	// Convertim rezultatul înapoi la 8 biți pentru afișare
	Mat result;
	workImg.convertTo(result, CV_8UC1);
	return result;
}
Mat ex3_7_draw_hue_quantization(Mat src_hsv) {
	Mat result = src_hsv.clone();

	vector<Mat> channels;
	split(src_hsv, channels);

	// Calculam PDF-ul canalului hue
	int* hist = ex3_1_compute_hist(channels[0]);
	float* pdf = ex3_2_compute_pdf(hist, channels[0].rows * channels[0].cols);

	// Determinam nuantele dominante pe canalul Hue
	grayscale_mapping map = ex3_5_1_multi_level_thresholding(5, 0.01f, pdf);

	// Aplicăm cuantizarea pe canalul Hue folosind nuantele dominante
	channels[0] = ex3_5_3_draw_multiple_thresholding(channels[0], map);

	// Combinăm canalele înapoi într-o imagine HSV
	merge(channels, result);
	cvtColor(result, result, COLOR_HSV2BGR);
	cvtColor(result, result, COLOR_BGR2GRAY);

	free(hist);
	free(pdf);
	if(map.grayscale_values)
		free(map.grayscale_values);

	return result;
}
Mat ex4_1_1_get_object_instance(const Mat& source, Vec3b color) {
	Mat result(source.rows, source.cols, CV_8UC1, Scalar(0));
	int th = 15; // Prag de toleranță

	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			Vec3b pixel = source.at<Vec3b>(i, j);

			// Verificăm dacă diferența pe fiecare canal (B, G, R) este mică
			if (abs(pixel[0] - color[0]) < th &&
				abs(pixel[1] - color[1]) < th &&
				abs(pixel[2] - color[2]) < th)
			{
				result.at<uchar>(i, j) = 255;
			}
		}
	}
	return result;
}
int ex4_1_2_compute_area(const Mat& binary_image) {
	int area = 0;
	for (int i = 0; i < binary_image.rows; i++) {
		for (int j = 0; j < binary_image.cols; j++) {
			if (binary_image.at<uchar>(i, j) > 0) {
				area++;
			}
		}
	}
	return area;
}
Point2d ex4_1_3_compute_center_of_mass(const Mat& binary_image) {
	double sum_x = 0;
	double sum_y = 0;
	int count = 0;
	for (int i = 0; i < binary_image.rows; i++) {
		for (int j = 0; j < binary_image.cols; j++) {
			if (binary_image.at<uchar>(i, j) > 0) {
				sum_x += j;
				sum_y += i;
				count++;
			}
		}
	}
	if (count == 0) return Point2d(0, 0); // Avoid division by zero
	return Point2d(sum_x / count, sum_y / count);
}
double ex4_1_4_compute_axis_of_elongation(const Mat& binary_image) {
	double mu20 = 0, mu02 = 0, mu11 = 0;
	Point2d center = ex4_1_3_compute_center_of_mass(binary_image);
	for (int i = 0; i < binary_image.rows; i++) {
		for (int j = 0; j < binary_image.cols; j++) {
			if (binary_image.at<uchar>(i, j) > 0) {
				double x = j - center.x;
				double y = i - center.y;
				mu20 += x * x;
				mu02 += y * y;
				mu11 += x * y;
			}
		}
	}
	double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
	return theta;
}
void ex4_1_5_draw_contour_points(const Mat& binary_image, Mat& canvas) {
	for (int i = 1; i < binary_image.rows - 1; i++) {
		for (int j = 1; j < binary_image.cols - 1; j++) {
			// Dacă pixelul curent este obiect, dar are un vecin fundal, e contur
			if (binary_image.at<uchar>(i, j) == 255) {
				if (binary_image.at<uchar>(i - 1, j) == 0 || binary_image.at<uchar>(i + 1, j) == 0 ||
					binary_image.at<uchar>(i, j - 1) == 0 || binary_image.at<uchar>(i, j + 1) == 0) {
					canvas.at<Vec3b>(i, j) = Vec3b(0, 255, 0); // Desenăm conturul cu verde
				}
			}
		}
	}
}
void ex4_1_6_draw_projections(const Mat& source, const Mat& binary_image, Mat& canvas) {
	// 1. Inițializăm vectorii de acumulare cu 0
	std::vector<int> h_proj(binary_image.rows, 0);
	std::vector<int> v_proj(binary_image.cols, 0);

	// 2. Parcurgem obiectul binar și numărăm pixelii de valoare 255 (alb)
	for (int i = 0; i < binary_image.rows; i++) {
		for (int j = 0; j < binary_image.cols; j++) {
			if (binary_image.at<uchar>(i, j) == 255) {
				h_proj[i]++; // Proiecția pe orizontală (rânduri)
				v_proj[j]++; // Proiecția pe verticală (coloane)
			}
		}
	}

	// 3. Creăm o imagine neagră pentru proiecții
	Mat projection_canvas = Mat::zeros(source.size(), CV_8UC3);

	// 4. Desenăm Proiecția Orizontală (linii galbene din stânga spre dreapta)
	for (int i = 0; i < h_proj.size(); i++) {
		line(projection_canvas, Point(0, i), Point(h_proj[i], i), Scalar(0, 255, 255));
	}

	// 5. Desenăm Proiecția Verticală (linii bleu de jos în sus)
	for (int j = 0; j < v_proj.size(); j++) {
		line(projection_canvas, Point(j, source.rows - 1), Point(j, source.rows - 1 - v_proj[j]), Scalar(255, 255, 0));
		
	}

	imshow("Projections", projection_canvas);
}
void ex4_1_7_geom_features(int event, int x, int y, int flags, void* param) {
	Mat* source = (Mat*)param;
	if (event != EVENT_LBUTTONDOWN) return;

	Vec3b color = source->at<Vec3b>(y, x);
	Mat object_instance = ex4_1_1_get_object_instance(*source, color);

	int area = ex4_1_2_compute_area(object_instance);
	Point2d center = ex4_1_3_compute_center_of_mass(object_instance);
	double phi = ex4_1_4_compute_axis_of_elongation(object_instance);

	// DACĂ ARIA ESTE PREA MARE, ÎNSEAMNĂ CĂ AM SELECTAT FUNDALUL
	// Într-un test rapid, dacă aria e mai mare de jumătate din imagine, inversăm
	if (area > (object_instance.rows * object_instance.cols) / 2) {
		bitwise_not(object_instance, object_instance);
		// Recalculăm trăsăturile pentru obiectul inversat
		area = ex4_1_2_compute_area(object_instance);
		center = ex4_1_3_compute_center_of_mass(object_instance);
		phi = ex4_1_4_compute_axis_of_elongation(object_instance);
	}

	std::cout << "Area: " << area << " | Center: " << center << " | Angle: " << phi << " rad" << std::endl;

	Mat canvas = source->clone();

	circle(canvas, center, 5, Scalar(0, 0, 255), -1);

	Point p1, p2;
	int len = 100; // lungime mai mare să fie vizibilă
	p1.x = center.x + len * cos(phi);
	p1.y = center.y + len * sin(phi);
	p2.x = center.x - len * cos(phi);
	p2.y = center.y - len * sin(phi);
	line(canvas, p1, p2, Scalar(255, 0, 0), 2);

	ex4_1_5_draw_contour_points(object_instance, canvas);
	ex4_1_6_draw_projections(*source, object_instance, canvas);

	imshow("Feature Detection", canvas);
}
void ex4_1_8_geom_features_window() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		namedWindow("My Window", 1);
		setMouseCallback("My Window", ex4_1_7_geom_features, &src);
		imshow("My Window", src);
		waitKey(0);
	}
}
void ex4_2_1_geom_features(int event, int x, int y, int flags, void* param) {
	if (event != EVENT_LBUTTONDOWN) return;

	// Recuperăm structura de date
	CallbackData* data = (CallbackData*)param;

	// 1. Identificăm culoarea și extragem obiectul
	Vec3b color = data->src.at<Vec3b>(y, x);
	Mat object_instance = ex4_1_1_get_object_instance(data->src, color);

	// 2. Calculăm trăsăturile folosind funcțiile tale anterioare
	int area = ex4_1_2_compute_area(object_instance);
	Point2d center = ex4_1_3_compute_center_of_mass(object_instance);
	double phi = ex4_1_4_compute_axis_of_elongation(object_instance); // radiani


	// DACĂ ARIA ESTE PREA MARE, ÎNSEAMNĂ CĂ AM SELECTAT FUNDALUL
	// Într-un test rapid, dacă aria e mai mare de jumătate din imagine, inversăm
	if (area > (object_instance.rows * object_instance.cols) / 2) {
		bitwise_not(object_instance, object_instance);
		// Recalculăm trăsăturile pentru obiectul inversat
		area = ex4_1_2_compute_area(object_instance);
		center = ex4_1_3_compute_center_of_mass(object_instance);
		phi = ex4_1_4_compute_axis_of_elongation(object_instance);
	}


	// Convertim phi în grade pentru a-l compara cu input-ul utilizatorului
	double phi_deg = phi * 180.0 / CV_PI;

	// 3. Verificăm condițiile de filtrare
	// Atenție: atan2 returnează în [-90, 90]. Asigură-te că intervalul phi_LOW/HIGH e logic.
	if (area < data->TH_area && phi_deg > data->phi_LOW && phi_deg < data->phi_HIGH) {

		Mat canvas = data->src.clone();

		// Desenăm centrul
		circle(canvas, center, 5, Scalar(0, 0, 255), -1);

		// Desenăm axa de alungire
		Point p1, p2;
		p1.x = center.x + 100 * cos(phi);
		p1.y = center.y + 100 * sin(phi);
		p2.x = center.x - 100 * cos(phi);
		p2.y = center.y - 100 * sin(phi);
		line(canvas, p1, p2, Scalar(255, 0, 0), 2);

		imshow("My Window", canvas);
		printf("Obiect VALID: Arie=%d, Unghi=%.2f\n", area, phi_deg);
	}
	else {
		printf("Obiect REPROBAT: Arie=%d, Unghi=%.2f\n", area, phi_deg);
	}
}
void ex4_2_2_geom_features_thresholding_window() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		CallbackData data;
		data.src = imread(fname);
		if (data.src.empty()) continue;

		// Citire parametri
		printf("Introduceti prag arie (TH_area): ");
		scanf("%d", &data.TH_area);
		printf("Introduceti phi_LOW (grade): ");
		scanf("%lf", &data.phi_LOW);
		printf("Introduceti phi_HIGH (grade): ");
		scanf("%lf", &data.phi_HIGH);

		namedWindow("My Window", 1);

		// Trimitem adresa structurii 'data' către callback
		setMouseCallback("My Window", ex4_2_1_geom_features, &data);

		imshow("My Window", data.src);
		waitKey(0);
	}
}
labels ex5_1_BFS(Mat src, const Point1* neighbours, int neighbours_size) {
	int height = src.rows;
	int width = src.cols;
	int tag = 0;

	// Inițializare matrice de etichete cu 0
	Mat tags = Mat::zeros(src.size(), CV_32SC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// Verificăm dacă pixelul este obiect (negru/0) și nevizitat (tag 0)
			if (src.at<uchar>(i, j) == 0 && tags.at<int>(i, j) == 0) {
				tag++;
				std::cout << "Componenta noua gasita la: " << i << "," << j << " (Tag: " << tag << ")" << std::endl;
				tags.at<int>(i, j) = tag;

				std::queue<Point1> Q;
				Q.push({ i, j });

				while (!Q.empty()) {
					Point1 p = Q.front();
					Q.pop();

					for (int k = 0; k < neighbours_size; k++) {
						int ni = p.x + neighbours[k].x;
						int nj = p.y + neighbours[k].y;

						if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
							if (src.at<uchar>(ni, nj) == 0 && tags.at<int>(ni, nj) == 0) {
								tags.at<int>(ni, nj) = tag;
								Q.push({ ni, nj });
							}
						}
					}
				}
			}
		}
	}

	// Vizualizare
	/*Mat dst = Mat::zeros(src.size(), CV_8UC3);
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			int label = tags.at<int>(r, c);
			if (label > 0) {
				// Folosim label-ul ca "seed" pentru a obține mereu aceeași culoare pentru același obiect
				RNG rng(label);
				// Generăm o culoare aleatorie, dar limităm luminozitatea să nu fie prea închisă
				dst.at<Vec3b>(r, c) = Vec3b(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
			}
			else {
				dst.at<Vec3b>(r, c) = Vec3b(255, 255, 255);
			}
		}
	}*/

	imshow("Original Image", src);
	//imshow("Connected Components", dst);
	return { tags, tag };
}
Mat ex5_2_color_labels(labels labels_str) {
	Mat label_img = labels_str.label;
	int no_labels = labels_str.no_labels;
	Mat result(label_img.rows, label_img.cols, CV_8UC3, Vec3b(255, 255, 255));

	std::vector<Vec3b> colors(no_labels + 1, Vec3b(255, 255, 255));
	std::default_random_engine gen;
	std::uniform_int_distribution<int> dis(0, 255);

	for (int i = 1; i <= no_labels; i++) {
		colors[i] = Vec3b(dis(gen), dis(gen), dis(gen));
	}

	for (int i = 0; i < label_img.rows; i++) {
		for (int j = 0; j < label_img.cols; j++) {
			int label = label_img.at<int>(i, j);
			if (label != 0)
				result.at<Vec3b>(i, j) = colors[label];
		}
	}
	return result;
}
int ex5_3_1_find(std::vector<int>& parent, int i) {
	while (parent[i] != i)
		i = parent[i];
	return i;
}
void ex5_3_2_unite(std::vector<int>& parent, int i, int j) {
	int root_i = ex5_3_1_find(parent, i);
	int root_j = ex5_3_1_find(parent, j);
	if (root_i != root_j) {
		if (root_i < root_j) parent[root_j] = root_i;
		else parent[root_i] = root_j;
	}
}
labels ex5_3_3_Two_pass_labeling(Mat source) {
	int rows = source.rows;
	int cols = source.cols;
	Mat label_img(rows, cols, CV_32SC1, Scalar(0));

	// Tabelul de echivalențe (inițial alocăm o dimensiune generoasă)
	std::vector<int> parent;
	parent.push_back(0); // Eticheta 0 este fundalul
	int current_label = 0;

	// --- PASUL 1: Etichetare temporară și găsirea echivalențelor ---
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			// Presupunem obiecte negre (0) pe fundal alb (255)
			if (source.at<uchar>(i, j) == 0) {
				std::vector<int> L; // Etichetele vecinilor parcurși deja (Sus și Stânga)

				// Verificăm vecinul de SUS
				if (i > 0 && label_img.at<int>(i - 1, j) > 0)
					L.push_back(label_img.at<int>(i - 1, j));
				// Verificăm vecinul din STÂNGA
				if (j > 0 && label_img.at<int>(i, j - 1) > 0)
					L.push_back(label_img.at<int>(i, j - 1));

				if (L.empty()) {
					// Caz 1: Niciun vecin etichetat -> Etichetă nouă
					current_label++;
					label_img.at<int>(i, j) = current_label;
					parent.push_back(current_label);
				}
				else {
					// Caz 2: Găsim vecini etichetați -> Luăm eticheta minimă
					int min_l = L[0];
					for (int label : L) if (label < min_l) min_l = label;
					label_img.at<int>(i, j) = min_l;

					// Înregistrăm echivalențele în tabel
					for (int label : L) {
						ex5_3_2_unite(parent, min_l, label);
					}
				}
			}
		}
	}
	Mat first_passing = ex5_2_color_labels({ label_img, current_label });
	imshow("intermediate", first_passing);

	// --- PASUL 2: Rezolvarea echivalențelor ---
	// Re-indexăm etichetele pentru a fi consecutive (1, 2, 3...)
	int final_label_count = 0;
	std::vector<int> new_labels(current_label + 1, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int lab = label_img.at<int>(i, j);
			if (lab > 0) {
				int root = ex5_3_1_find(parent, lab);

				// Opțional: re-mapare pentru a avea etichete consecutive
				if (new_labels[root] == 0) {
					final_label_count++;
					new_labels[root] = final_label_count;
				}
				label_img.at<int>(i, j) = new_labels[root];
			}
		}
	}

	return { label_img, final_label_count };
}
void ex6_3_1_reconstruct_contour(const char* filename, Mat& background) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		printf("Eroare la deschiderea fisierului %s\n", filename);
		return;
	}

	int startRow, startCol, numPoints;
	// Citim prima linie: punctul de start (53 161 in cazul tau)
	if (!(file >> startRow >> startCol)) return;
	// Citim a doua linie: numarul total de puncte (302)
	if (!(file >> numPoints)) return;

	Point1 p_curr = { startRow, startCol };

	// Matrice pentru rezultat (convertim la BGR pentru a desena cu roșu)
	Mat dst;
	if (background.channels() == 1)
		cvtColor(background, dst, COLOR_GRAY2BGR);
	else
		dst = background.clone();

	// Desenăm punctul de start
	dst.at<Vec3b>(p_curr.x, p_curr.y) = Vec3b(0, 0, 255);

	int code;
	// Citim restul codurilor din fisier
	while (file >> code) {
		if (code >= 0 && code <= 7) {
			// Actualizăm poziția conform codului
			p_curr.x += freeman_neighbors[code].x;
			p_curr.y += freeman_neighbors[code].y;

			// Verificăm limitele imaginii pentru siguranță
			if (p_curr.x >= 0 && p_curr.x < dst.rows && p_curr.y >= 0 && p_curr.y < dst.cols) {
				dst.at<Vec3b>(p_curr.x, p_curr.y) = Vec3b(0, 0, 255); // Desenăm cu roșu
			}
		}
	}

	file.close();
	imshow("Contur Reconstruit: EXCELLENT", dst);
}
void ex6_3_reconstructFromChain() {
	// Încarcă imaginea de fundal
	Mat background = imread("D:/FACULTATE/SEM II/PI/LAB/L06/gray_background.bmp", IMREAD_GRAYSCALE);
	if (background.empty()) {
		// Dacă nu găsești imaginea, creează una gri pentru test
		background = Mat(256, 512, CV_8UC1, Scalar(128));
	}

	ex6_3_1_reconstruct_contour("D:/FACULTATE/SEM II/PI/LAB/L06/reconstruct.txt", background);
	waitKey(0);
}
ChainResults ex6_2_1_compute_chain_codes(Mat src, Point1 p0) {
	ChainResults res;
	Point1 p_current = p0;
	int dir = 0; // Direcția inițială
	bool finished = false;

	// --- Pasul 1: Generarea Codului Înlănțuit ---
	while (!finished) {
		bool found = false;
		// Căutăm circular începând de la ultima direcție cunoscută
		int search_start = (dir + 5) % 8;

		for (int i = 0; i < 8; i++) {
			int k = (search_start + i) % 8;
			int ni = p_current.x + freeman_neighbors[k].x;
			int nj = p_current.y + freeman_neighbors[k].y;

			if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
				if (src.at<uchar>(ni, nj) == 0) { // Presupunem obiect negru
					res.chainCode.push_back(k); // Salvăm codul direcției
					p_current = { ni, nj };
					dir = k;
					found = true;
					break;
				}
			}
		}

		// Ne oprim când revenim la punctul de start
		if (p_current.x == p0.x && p_current.y == p0.y) {
			finished = true;
		}
		if (!found) break;
	}

	// --- Pasul 2: Calcularea Derivatei ---
	// Formula: d_i = (c_i - c_{i-1}) mod 8 (unde c este codul curent)
	int n = res.chainCode.size();
	if (n > 0) {
		for (int i = 0; i < n; i++) {
			int prev = (i == 0) ? res.chainCode[n - 1] : res.chainCode[i - 1];
			int diff = (res.chainCode[i] - prev + 8) % 8;
			res.derivative.push_back(diff);
		}
	}

	return res;
}
void ex6_2_chainCodeAnalysis() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		// 1. Citire imagine (neapărat Grayscale)
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) continue;

		// 2. Binarizare - Obiectul trebuie să fie NEGRU (0) și fundalul ALB (255)
		// Dacă imaginea ta are obiecte albe, folosește THRESH_BINARY_INV
		Mat binary;
		threshold(src, binary, 127, 255, THRESH_BINARY);

		// 3. Găsire punct de start P0
		Point1 p0 = findPo(binary);

		if (p0.x != -1) {
			// 4. Calculare Cod Inlănțuit și Derivată
			ChainResults res = ex6_2_1_compute_chain_codes(binary, p0);

			// 5. Afișare în consolă
			printf("\n--- ANALIZA OBIECT ---\n");
			printf("Punct de start P0: (%d, %d)\n", p0.x, p0.y);
			printf("Numar de puncte contur: %zu\n", res.chainCode.size());

			printf("\nCodul Inlantuit (Freeman):\n");
			for (int val : res.chainCode) {
				printf("%d", val);
			}

			printf("\n\nDerivata Codului Inlantuit:\n");
			for (int val : res.derivative) {
				printf("%d", val);
			}
			printf("\n----------------------\n");

			// 6. Vizualizare (opțional, folosind funcția de desenare contur)
			contour cnt;
			// Reconstruim un obiect de tip contour pentru a-l desena
			Point1 p_curr = p0;
			cnt.border.push_back(p0);
			for (int code : res.chainCode) {
				p_curr.x += freeman_neighbors[code].x;
				p_curr.y += freeman_neighbors[code].y;
				cnt.border.push_back(p_curr);
			}

			Mat dst = draw_contour(cnt, src);
			imshow("Contur Analizat", dst);
		}
		else {
			printf("Nu s-a gasit niciun obiect in imagine.\n");
		}

		waitKey(0);
		destroyAllWindows();
	}
}
contour ex6_1_1_trace_contour(Mat src, Point1 p0) {
	contour cnt;
	Point1 p_current = p0;

	// Stabilim prima direcție de căutare
	int dir = 7;
	int first_dir = -1; // Vom salva prima direcție de succes aici

	bool finished = false;
	while (!finished) {
		bool found = false;
		int search_dir = (dir + 6) % 8; // Moore Neighborhood: înapoi cu 2 poziții

		for (int i = 0; i < 8; i++) {
			int k = (search_dir + i) % 8;
			int ni = p_current.x + neighbours8[k].x;
			int nj = p_current.y + neighbours8[k].y;

			if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
				// VERIFICĂ: Dacă obiectele tale sunt NEGRE, lasă 0. 
				// Dacă sunt ALBE (după threshold-ul tău de 127), pune 255!
				if (src.at<uchar>(ni, nj) == 0) {
					Point1 next_p = { ni, nj };

					// Adăugăm punctul DOAR dacă nu am revenit la start
					if (next_p.x == p0.x && next_p.y == p0.y && cnt.border.size() > 1) {
						return cnt; // Am închis conturul
					}

					cnt.border.push_back(next_p);
					p_current = next_p;
					dir = k;
					found = true;
					break;
				}
			}
		}

		// CRITERIUL DE OPRIRE: Ne-am întors la start cu aceeași direcție
		if (p_current.x == p0.x && p_current.y == p0.y) {
			// Dacă folosim Criteriul lui Jacob, verificăm și direcția.
			// Pentru forme simple, e suficient să verificăm dacă am făcut măcar 2 pași
			if (cnt.border.size() > 1) {
				finished = true;
			}
		}

		if (!found || cnt.border.size() > (src.rows * src.cols)) break;
	}

	return cnt;
}
void ex6_1_contourTracing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) continue;

		// Binarizare pentru siguranță (0 = obiect, 255 = fundal)
		threshold(src, src, 127, 255, THRESH_BINARY);

		Point1 p0 = findPo(src);

		if (p0.x != -1) {
			// 1. Urmărim conturul
			contour cnt = ex6_1_1_trace_contour(src, p0);

			// 2. Desenăm conturul folosind funcția ta
			Mat dst = draw_contour(cnt, src);

			imshow("Contur Detectat", dst);
			printf("Contur gasit! Numar puncte: %zu\n", cnt.border.size());
		}
		else {
			printf("Nu a fost gasit niciun obiect.\n");
		}
		waitKey();
	}
}
Mat ex7_1_1_erosion(Mat src) {
	Mat dst = Mat::zeros(src.size(), CV_8UC1);

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			bool foundWhite = false;
			for (int ky = -1; ky <= 1; ky++) {
				for (int kx = -1; kx <= 1; kx++) {
					if (src.at<uchar>(i + ky, j + kx) == 255) {
						foundWhite = true;
						break;
					}
				}
				if (foundWhite) break;
			}
			dst.at<uchar>(i, j) = foundWhite ? 255 : 0;
		}
	}
	return dst;
}
Mat ex7_1_2_dilation(const Mat& src) {
	Mat dst = Mat::zeros(src.size(), CV_8UC1);

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			bool fits = true;
			for (int ky = -1; ky <= 1; ky++) {
				for (int kx = -1; kx <= 1; kx++) {
					if (src.at<uchar>(i + ky, j + kx) == 0) {
						fits = false;
						break;
					}
				}
				if (!fits) break;
			}
			if (fits) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}
Mat ex7_1_3_opening(Mat src) {
	Mat eroded, opened;
	eroded = ex7_1_1_erosion(src);
	opened = ex7_1_2_dilation(eroded);
	return opened;
}
Mat ex7_1_4_closing(Mat src) {
	Mat dilated, closed;
	dilated = ex7_1_2_dilation(src);
	closed = ex7_1_1_erosion(dilated);
	return closed;
}
Mat ex7_4_1_region_filling(Mat src, Point seed) {
	// 1. Pregătim masca (bariera)
	// Trebuie să fie NEGRU unde e conturul și ALB unde se poate extinde
	Mat mask = src.clone();

	// 2. IMPORTANT: Închidem marginile imaginii cu o linie neagră
	// Asta oprește "scurgerea" dacă obiectul atinge marginea ferestrei
	line(mask, Point(0, 0), Point(mask.cols - 1, 0), Scalar(0), 1);
	line(mask, Point(0, mask.rows - 1), Point(mask.cols - 1, mask.rows - 1), Scalar(0), 1);
	line(mask, Point(0, 0), Point(0, mask.rows - 1), Scalar(0), 1);
	line(mask, Point(mask.cols - 1, 0), Point(mask.cols - 1, mask.rows - 1), Scalar(0), 1);

	// 3. Inițializăm X (interiorul care va fi umplut)
	Mat X = Mat::zeros(src.size(), CV_8UC1);

	// Verificăm dacă seed-ul este valid (în interiorul imaginii și pe un pixel alb)
	if (seed.x >= 0 && seed.x < src.cols && seed.y >= 0 && seed.y < src.rows) {
		if (src.at<uchar>(seed.y, seed.x) == 255) { // Doar dacă e pe alb
			X.at<uchar>(seed.y, seed.x) = 255;
		}
		else {
			printf("Eroare: Punctul de start a picat pe contur sau in exterior!\n");
			return src;
		}
	}

	// 4. Element structurant de tip CRUCE (vecinătate de 4)
	// Este obligatoriu pentru a nu "sări" prin colțurile contururilor subțiri
	Mat B = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);

	Mat X_prev;
	int max_it = 10000; // Siguranță împotriva buclelor infinite

	while (max_it--) {
		X.copyTo(X_prev);

		dilate(X, X, B); // Creștem regiunea
		bitwise_and(X, mask, X); // Limităm creșterea la spațiul alb delimitat de negru

		// Dacă nu mai sunt schimbări, am terminat
		if (countNonZero(X != X_prev) == 0) break;
	}

	// 5. RECONSTRUCȚIA VIZUALĂ:
	// Până acum, X este interiorul (alb) pe fundal negru.
	// Noi vrem interiorul negru pe fundal alb (ca în original).

	Mat result;
	bitwise_not(X, result); // Inversăm X: acum interiorul e negru, restul e alb

	// Combinăm cu conturul original pentru a păstra detaliile
	bitwise_and(result, src, result);

	return result;
}
void ex7_4_region_filling() {
	char fname3[MAX_PATH];
	while (openFileDlg(fname3))
	{
		Mat src = imread(fname3, IMREAD_GRAYSCALE);
		threshold(src, src, 127, 255, THRESH_BINARY);

		// Strategie automată: Inversăm și folosim FloodFill pe margini
		// pentru a izola doar "golurile" interioare
		Mat temp = src.clone();
		// Umplem exteriorul pornind de la (0,0) cu o altă valoare
		floodFill(temp, Point(0, 0), Scalar(128));

		// Pixelii care au rămas 255 (albi) sunt acum garantat în interiorul formelor închise
		Mat seeds;
		inRange(temp, 255, 255, seeds);

		// Găsim primul pixel alb din 'seeds' pentru a porni Region Filling-ul tău
		Point start_point(-1, -1);
		for (int i = 0; i < seeds.rows; i++) {
			for (int j = 0; j < seeds.cols; j++) {
				if (seeds.at<uchar>(i, j) == 255) {
					start_point = Point(j, i);
					break;
				}
			}
			if (start_point.x != -1) break;
		}

		if (start_point.x != -1) {
			Mat filled = ex7_4_1_region_filling(src, start_point);
			//bitwise_not(filled, filled);
			imshow("Rezultat automat", filled);
			imshow("Imagine originala", src);
			waitKey(0);
		}
	}
}
void ex8_1_processing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;
		double n = (double)height * width;

		// 1. Calculul Mediei
		double sum = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sum += img.at<uchar>(i, j);
			}
		}
		double mean = sum / n;

		// 2. Calculul Deviatiei Standard
		double sum_sq_diff = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double diff = img.at<uchar>(i, j) - mean;
				sum_sq_diff += diff * diff;
			}
		}
		double std_dev = sqrt(sum_sq_diff / n);

		printf("Media intensitatii: %lf\n", mean);
		printf("Deviatia standard: %lf\n", std_dev);

		// 3. Calculul Histogramei
		int* hist = (int*)calloc(256, sizeof(int));
		hist = ex3_1_compute_hist(img);

		showHistogram("Histograma Intensitati", hist, 256, 200);

		// 4. Calculul Histogramei Cumulative
		int hist_cum[256] = { 0 };
		hist_cum[0] = hist[0];
		for (int i = 1; i < 256; i++) {
			hist_cum[i] = hist_cum[i - 1] + hist[i];
		}
		showHistogram("Histograma Cumulativa", hist_cum, 256, 200);

		imshow("Imagine Originala", img);
		waitKey(0);
	}
}
void ex8_2_automaticThresholding() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		if (img.empty()) continue;

		int height = img.rows;
		int width = img.cols;

		// 1. Calculul histogramei
		int hist[256] = { 0 };
		int min_val = 255, max_val = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar pixel = img.at<uchar>(i, j);
				hist[pixel]++;
				if (pixel < min_val) min_val = pixel;
				if (pixel > max_val) max_val = pixel;
			}
		}

		// 2. Initializarea pragului T cu media dintre intensitatea minima si maxima
		float T = (float)(min_val + max_val) / 2.0f;
		float T_vechi;
		float epsilon = 0.5f; // Eroarea

		// 3. Iteratie pentru determinarea pragului optim
		do {
			T_vechi = T;

			double sum_low = 0, count_low = 0;
			double sum_high = 0, count_high = 0;

			// Calculam mediile celor doua grupuri separate de pragul curent T
			for (int i = 0; i < 256; i++) {
				if (i <= T) {
					sum_low += (double)i * hist[i];
					count_low += hist[i];
				}
				else {
					sum_high += (double)i * hist[i];
					count_high += hist[i];
				}
			}

			float mu1 = (count_low > 0) ? (float)(sum_low / count_low) : 0;
			float mu2 = (count_high > 0) ? (float)(sum_high / count_high) : 0;

			T = (mu1 + mu2) / 2.0f;

		} while (abs(T - T_vechi) > epsilon);

		int finalThreshold = (int)T;
		printf("Pragul de binarizare determinat automat: %d\n", finalThreshold);

		// 4. Binarizarea imaginii folosind pragul obtinut
		Mat binaryImg(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				binaryImg.at<uchar>(i, j) = (img.at<uchar>(i, j) > finalThreshold) ? 255 : 0;
			}
		}

		imshow("Original", img);
		imshow("Binarizata (Prag Automat)", binaryImg);

		showHistogram("Histograma", hist, 256, 200);

		waitKey(0);
	}
}
uchar clamp(double val) {
	if (val < 0) return 0;
	if (val > 255) return 255;
	return (uchar)val;
}
void ex8_3_histogramTransformations() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) continue;

		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		uchar LUT[256];

		printf("\nAlege transformarea:\n1. Negativ\n2. Latire/Ingustare\n3. Corectie Gamma\n4. Luminozitate\n");
		int option;
		scanf("%d", &option);

		if (option == 1) { // Negativ
			for (int i = 0; i < 256; i++) LUT[i] = 255 - i;
		}
		else if (option == 2) { // Latire/Ingustare
			int goutMin, goutMax;
			int ginMin = 255, ginMax = 0;
			// Gasim minim/maxim in imaginea sursa
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					uchar p = src.at<uchar>(i, j);
					if (p < ginMin) ginMin = p;
					if (p > ginMax) ginMax = p;
				}
			}
			printf("Introduceti goutMIN si goutMAX: ");
			scanf("%d %d", &goutMin, &goutMax);

			for (int i = 0; i < 256; i++) {
				if (i <= ginMin) LUT[i] = goutMin;
				else if (i >= ginMax) LUT[i] = goutMax;
				else LUT[i] = goutMin + (i - ginMin) * (goutMax - goutMin) / (ginMax - ginMin);
			}
		}
		else if (option == 3) { // Gamma
			double gamma;
			printf("Introduceti coeficientul Gamma: ");
			scanf("%lf", &gamma);
			for (int i = 0; i < 256; i++) {
				// Normalizam la [0, 1], aplicam puterea, revenim la [0, 255]
				LUT[i] = clamp(255.0 * pow((double)i / 255.0, gamma));
			}
		}
		else if (option == 4) { // Luminozitate
			int offset;
			printf("Introduceti valoarea de crestere (poate fi negativa): ");
			scanf("%d", &offset);
			for (int i = 0; i < 256; i++) LUT[i] = clamp(i + offset);
		}

		// Aplicare LUT pe imagine
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = LUT[src.at<uchar>(i, j)];
			}
		}

		// Afisare rezultate
		imshow("Sursa", src);
		imshow("Destinatie", dst);
		showHistogram("Histograma Sursa", ex3_1_compute_hist(src), src.cols, src.rows);
		showHistogram("Histograma Destinatie", ex3_1_compute_hist(dst), dst.cols, dst.rows);

		waitKey(0);
		destroyAllWindows();
	}
}

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}
void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}
void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}
void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}
void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}
void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}
void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}
void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}
void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}
void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}
void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			printf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}
void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}
void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


/*int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		{
			printf("Menu:\n");
			printf(" 1 - Open image\n");
			printf(" 2 - Open BMP images from folder\n");
			printf(" 3 - Image negative\n");
			printf(" 4 - Image negative (fast)\n");
			printf(" 5 - BGR->Gray\n");
			printf(" 6 - BGR->Gray (fast, save result to disk) \n");
			printf(" 7 - BGR->HSV\n");
			printf(" 8 - Resize image\n");
			printf(" 9 - Canny edge detection\n");
			printf(" 10 - Edges in a video sequence\n");
			printf(" 11 - Snap frame from live video\n");
			printf(" 12 - Mouse callback demo\n");
			printf(" 1_3 - (change gray levels - aditive) - 13\n");
			printf(" 1_4 - (change gray levels - multiplicative) - 14\n");
			printf(" 1_5 - (4 coloured square - 256x256) - 15\n");
			printf(" 1_6 - (inverse of a matrix) - 16\n");
			printf(" 2_1 - (extract RGB channels) - 17\n");
			printf(" 2_2 - (from RGB to Grayscale) - 18\n");
			printf(" 2_3 - (from Grayscale to Black and White) - 19\n");
			printf(" 2_4 - (RGB to HSV) - 20\n");
			printf(" 2_5 - (check if pixel location is valid) - 21\n");
			printf(" 3_1 - (Histogram equalization) - 22\n");
			printf(" 3_2 - (FDP) - 23\n");
			printf(" 3_4 - (adaptive histogram equalization) - 24\n");
			printf(" 3_5 - (gray-level reduction / multi-thresholding) - 35\n");
			printf(" 3_6 - (Floyd-Steinberg Dithering) - 36\n");
			printf(" 3_7 - (HSV Hue quantization) - 37\n");
			printf(" 4_1 - (geometric features) - 26\n");
			printf(" 4_2 - (geometric features - thresholding) - 38\n");
			printf(" 5_1 - (BFS)  - 27\n");
			printf(" 5_2 - (Two pass coloring) - 39\n");
			printf(" 6_1 - (contour tracing) - 28\n");
			printf(" 6_2 - (chain code analysis) - 40\n");
			printf(" 6_3 - (reconstruct contour from chain code) - 41\n");
			printf(" 7_1 - (erosion) - 29\n");
			printf(" 7_2 - (dilation) - 30\n");
			printf(" 7_3 - (close) - 31\n");
			printf(" 7_4 - (open) - 32\n");
			printf(" 7_5 - (contour tracing) - 33\n");
			printf(" 7_6 - (region filling) - 42\n");
			printf(" 8_1 - (intensity properties of image) - 43\n");
			printf(" 8_2 - (automatic tresholding) - 44\n");
			printf(" 8_3 - (histogram transformations) - 45\n");
			printf(" 0 - Exit\n\n");
		}
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1: testOpenImage(); break;
			case 2: testOpenImagesFld(); break;
			case 3: testNegativeImage(); break;
			case 4: testNegativeImageFast(); break;
			case 5: testColor2Gray(); break;
			case 6: testImageOpenAndSave(); break;
			case 7: testBGR2HSV(); break;
			case 8: testResize(); break;
			case 9: testCanny(); break;
			case 10: testVideoSequence(); break;
			case 11: testSnap(); break;
			case 12: testMouseClick(); break;
			case 13: ex1_3_change_gray_levels_additive(); break;
			case 14: ex1_4_change_gray_levels_multiplicative(); break;
			case 15: ex1_5_four_coloured_square(); break;
			case 16: ex1_6_inverse_of_matrix(); break;
			case 17: ex2_1_extract_RGB_channels(); break;
			case 18: testColor2Gray(); break;
			case 19: ex2_3_grayscale_to_BW(); break;
			case 20: ex2_4_RBG_to_HSV(); break;
			case 22:
			{
				char fname[MAX_PATH];
				while (openFileDlg(fname))
				{
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					int* hist = ex3_1_compute_hist(src);
					showHistogram("hist", hist, 256, 200);
					waitKey();
					free(hist);
				}
				break;
			}
			case 23:
			{
				char fname2[MAX_PATH];
				while (openFileDlg(fname2))
				{
					Mat src = imread(fname2, IMREAD_GRAYSCALE);
					int* hist = ex3_1_compute_hist(src);
					float* pdf = ex3_2_compute_pdf(hist, src.rows * src.cols);
					for (int i = 0; i < 256; i++)
						printf("pdf[%d] = %.3f\n", i, pdf[i]);

					for (int i = 0; i < 256; i += 8) { // Sărim din 8 în 8 ca să nu umplem ecranul
						printf("%3d | ", i);
						int barLength = (int)(pdf[i] * 1000); // Scalăm probabilitatea pentru vizualizare
						for (int k = 0; k < barLength; k++) printf("*");
						printf("\n");
					}
					waitKey();
					free(hist);
					free(pdf);
				}
				break;
			}
			case 24:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					int no_bins;
					printf("Number of bins: ");
					scanf("%d", &no_bins);
					int* hist = ex3_4_compute_hist_custom(src, no_bins);
					showHistogram("hist_custom", hist, no_bins, 200);
					waitKey();
					free(hist);
				}
				break;
			}
			case 25: break;
			case 26: ex4_1_8_geom_features_window(); break;
			case 27:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					// Înainte de buclele for, asigură-te că obiectele sunt clar delimitate
					labels current_labels = ex5_1_BFS(src, neighbours8, 8);
					Mat colored = ex5_2_color_labels(current_labels);
					imshow("Final Colored Result", colored);
					printf("Number of connected components: %d\n", current_labels.no_labels);
					waitKey(0);
				}
				break;
			}
			case 28:
			{
				ex6_1_contourTracing();
				break;
			}
			case 29:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					if (src.empty()) printf("eroare");
					Mat dst;
					int n = 3;

					Mat tempSrc = src.clone();
					Mat tempDst;

					for (int i = 0; i < n; i++) {
						tempDst = ex7_1_1_erosion(tempSrc);
						tempSrc = tempDst.clone();
					}
					dst = tempDst;
					imshow("Eroziune repetata", dst);
					imshow("Imagine Originala", src);
				}
				break;
			}
			case 30:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					if (src.empty()) {
						printf("Eroare la incarcarea imaginii!\n");
						continue;
					}

					int n = 3;
					Mat tempSrc = src.clone();
					Mat tempDst;

					for (int i = 0; i < n; i++) {
						tempDst = ex7_1_2_dilation(tempSrc);
						tempSrc = tempDst.clone(); 
					}

					imshow("Dilatatie repetata", tempDst);
				    imshow("Imagine Originala", src);

					waitKey(0); 
				}
			}
			case 31:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					if (src.empty()) {
						printf("Eroare la incarcarea imaginii!\n");
						continue;
					}
					Mat closed = ex7_1_4_closing(src);
					imshow("Imagine inchisa", closed);
					imshow("Imagine Originala", src);
					waitKey(0); 
				}
				break;
			}
			case 32:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					if (src.empty()) {
						printf("Eroare la incarcarea imaginii!\n");
						continue;
					}
					Mat opened = ex7_1_3_opening(src);
					imshow("Imagine deschisa", opened);
					imshow("Imagine Originala", src);
					waitKey(0); 
				}
				break;
			}
			case 33:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					threshold(src, src, 127, 255, THRESH_BINARY);
					Mat imgErodata = ex7_1_1_erosion(src);
					Mat contur = Mat::zeros(src.size(), CV_8UC1);

					for (int i = 0; i < src.rows; i++) {
						for (int j = 0; j < src.cols; j++) {
							// Pixelul este de contur dacă în original era fundal (255) 
							// dar în versiunea erodată a obiectului a devenit obiect (0)
							// SAU mai simplu: diferența absolută
							uchar valOriginal = src.at<uchar>(i, j);
							uchar valErodat = imgErodata.at<uchar>(i, j);

							// Dacă sunt diferite, înseamnă că pixelul aparține marginii
							if (valOriginal != valErodat) {
								contur.at<uchar>(i, j) = 0; // Desenăm conturul cu negru
							}
							else {
								contur.at<uchar>(i, j) = 255; // Restul rămâne alb
							}
						}
					}
					imshow("Imagine Originala", src);
					imshow("Contur", contur);
				}
				break;
			}
			case 34:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					threshold(src, src, 127, 255, THRESH_BINARY);
					Mat temp = ex7_1_2_dilation(src);
					Mat rezultatUmplere = ex7_1_1_erosion(temp);
					imshow("Goluri umplute (Closing)", rezultatUmplere);
				}
				break;
			}
			case 35:
			{
				char fname[MAX_PATH];
				while (openFileDlg(fname))
				{
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					if (src.empty()) continue;

					// 1. Calculăm histograma și PDF-ul (necesare pentru multi_level_thresholding)
					int* hist = ex3_1_compute_hist(src); // Funcția ta de la ex 3.1
					float* pdf = ex3_2_compute_pdf(hist, src.rows * src.cols); // Funcția ta de la ex 3.2

					// 2. Parametrii pentru detecția maximelor
					int wh = 5;       // Window half-width (mărimea ferestrei de căutare)
					float th = 0.0001f; // Threshold (pragul de relevanță pentru vârfuri)

					// 3. Apelăm funcția care găsește nuanțele dominante
					grayscale_mapping mapping = ex3_5_1_multi_level_thresholding(wh, th, pdf);

					// 4. Aplicăm reducerea pe imagine
					Mat dst = ex3_5_3_draw_multiple_thresholding(src, mapping);

					// Afișăm rezultatele
					printf("S-au gasit %d niveluri de gri dominante.\n", mapping.count_grayscale_values);
					imshow("Original", src);
					imshow("Reducere Niveluri (Multi-thresholding)", dst);

					waitKey();

					// 5. Eliberăm memoria (foarte important!)
					free(hist);
					free(pdf);
					free(mapping.grayscale_values); // Eliberăm malloc-ul din structură
				}
				break;
			}
			case 36: 
			{
				char fname[MAX_PATH];
				while (openFileDlg(fname))
				{
					// 1. Încărcăm imaginea originală (Grayscale)
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					if (src.empty()) continue;

					// 2. Pregătim datele necesare pentru a găsi nuanțele dominante
					int* hist = ex3_1_compute_hist(src); // Funcția ta de calcul histogramă
					float* pdf = ex3_2_compute_pdf(hist, src.rows * src.cols); // Funcția ta de PDF

					// 3. Calculăm maparea (vârfurile histogramei)
					// wh = fereastra, th = pragul de detecție
					int wh = 5;
					float th = 0.0001f;
					grayscale_mapping mapping = ex3_5_1_multi_level_thresholding(wh, th, pdf);

					// 4. APELUL FUNCȚIEI TALE PRINCIPALEn
					// Aceasta va apela intern process_pixel și diffuse_error
					Mat dst = ex3_6_3_draw_floyd_steinberg(src, mapping);

					// 5. Afișare rezultate
					imshow("Original", src);
					imshow("Floyd-Steinberg Dithering", dst);

					printf("S-au folosit %d niveluri de gri pentru dithering.\n", mapping.count_grayscale_values);

					waitKey();

					// 6. CURĂȚENIE (Foarte important pentru a evita memory leaks)
					free(hist);
					free(pdf);
					free(mapping.grayscale_values); // Eliberăm memoria alocată în multi_level_thresholding
				}
				break;
			}
			case 37: 
			{ 
				char fname[MAX_PATH];
				while (openFileDlg(fname))
				{
					Mat src = imread(fname);
					if (src.empty()) continue;

					// Apelăm funcția: wh=5, th=0.0003, setMaxSV=true (pentru culori neon)
					Mat dst = ex3_7_draw_hue_quantization(src);

					imshow("Original", src);
					imshow("Hue Quantized Result", dst);
					waitKey();
				}
				break;
			}
			case 38:
			{
				ex4_2_2_geom_features_thresholding_window();
				break;
			}
			case 39:
			{
				char fname3[MAX_PATH];
				while (openFileDlg(fname3))
				{
					Mat src = imread(fname3, IMREAD_GRAYSCALE);
					labels current_labels = ex5_3_3_Two_pass_labeling(src);
					Mat colored = ex5_2_color_labels(current_labels);
					imshow("Two Pass Coloring", colored);
					printf("Number of connected components: %d\n", current_labels.no_labels);
					waitKey(0);
				}
				break;
			}
			case 40:
			{
				ex6_2_chainCodeAnalysis();
				break;
			}
			case 41:
			{
				ex6_3_reconstructFromChain();
				break;
			}
			case 42: { ex7_4_region_filling(); break; }
			case 43: {
				ex8_1_processing();
				break;
			}
			case 44: 
			{
				ex8_2_automaticThresholding();
				break;
			}
			case 45:
			{
				ex8_3_histogramTransformations();
				break;
			}
			case 21:
			{
				int x = 0, y = 0;
				scanf("%d", &x);
				scanf("%d", &y);
				isInside(x, y);
				break;
			}
			
		}
	}
	while (op!=0);
	return 0;
}*/