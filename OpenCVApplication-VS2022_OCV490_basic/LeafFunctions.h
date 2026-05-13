#pragma once
#include "stdafx.h"
#include "common.h" // Or whatever includes Mat and Point
#include <vector>
#include <string>
#include <iostream>

using namespace std;

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

using namespace std;
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

Point1 findPo(Mat src) ;
void display_codes(ChainResults res);
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height);
Mat draw_contour(contour cnt, Mat src);

void ex1_3_change_gray_levels_additive();
void ex1_4_change_gray_levels_multiplicative();
void ex1_5_four_coloured_square();
void ex1_6_inverse_of_matrix();
void ex2_1_extract_RGB_channels();
void ex2_3_grayscale_to_BW();
void isInside(int x, int y);

void ex2_4_RBG_to_HSV() ;
int* ex3_1_compute_hist(Mat img);
float* ex3_2_compute_pdf(int* hist, int totalPixels);
int* ex3_4_compute_hist_custom(Mat img, int no_bins);
grayscale_mapping ex3_5_1_multi_level_thresholding(int wh, float th, float* pdf);
uchar ex3_5_2_find_closest_histogram_maximum(uchar old_pixel, grayscale_mapping map);
Mat ex3_5_3_draw_multiple_thresholding(Mat source, grayscale_mapping map) ;
void ex3_6_1_diffuse_error(Mat& img, int r, int c, float error) ;
float ex3_6_2_process_pixel_floyd(Mat& img, int r, int c, grayscale_mapping map) ;
Mat ex3_6_3_draw_floyd_steinberg(Mat source, grayscale_mapping map) ;
Mat ex3_7_draw_hue_quantization(Mat src_hsv) ;
Mat ex4_1_1_get_object_instance(const Mat& source, Vec3b color) ;
int ex4_1_2_compute_area(const Mat& binary_image) ;
Point2d ex4_1_3_compute_center_of_mass(const Mat& binary_image) ;
double ex4_1_4_compute_axis_of_elongation(const Mat& binary_image) ;
void ex4_1_5_draw_contour_points(const Mat& binary_image, Mat& canvas) ;
void ex4_1_6_draw_projections(const Mat& source, const Mat& binary_image, Mat& canvas) ;
void ex4_1_7_geom_features(int event, int x, int y, int flags, void* param) ;
void ex4_1_8_geom_features_window() ;
void ex4_2_1_geom_features(int event, int x, int y, int flags, void* param) ;
void ex4_2_2_geom_features_thresholding_window() ;
labels ex5_1_BFS(Mat src, const Point1* neighbours, int neighbours_size) ;
Mat ex5_2_color_labels(labels labels_str) ;
int ex5_3_1_find(std::vector<int>& parent, int i) ;
void ex5_3_2_unite(std::vector<int>& parent, int i, int j) ;
labels ex5_3_3_Two_pass_labeling(Mat source) ;
void ex6_3_1_reconstruct_contour(const char* filename, Mat& background) ;
void ex6_3_reconstructFromChain() ;
ChainResults ex6_2_1_compute_chain_codes(Mat src, Point1 p0) ;;
void ex6_2_chainCodeAnalysis() ;
contour ex6_1_1_trace_contour(Mat src, Point1 p0) ;;
void ex6_1_contourTracing() ;
Mat ex7_1_1_erosion(Mat src) ;
Mat ex7_1_2_dilation(const Mat& src) ;
Mat ex7_1_3_opening(Mat src) ;
Mat ex7_1_4_closing(Mat src) ;
Mat ex7_4_1_region_filling(Mat src, Point seed) ;
void ex7_4_region_filling() ;
void ex8_1_processing() ;
void ex8_2_automaticThresholding() ;;
uchar clamp(double val) ;
void ex8_3_histogramTransformations() ;

void testOpenImage();
void testOpenImagesFld();
void testImageOpenAndSave();

void testNegativeImage();
void testNegativeImageFast();
void testColor2Gray();
void testBGR2HSV();
void testResize();
void testCanny();

void testVideoSequence();
void testSnap();
void MyCallBackFunc(int event, int x, int y, int flags, void* param);
void testMouseClick();