#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>
#include <fstream>
#include "LeafFunctions.h" 

using namespace std;
using namespace cv;

extern wchar_t* projectPath;
#include <filesystem>
#include <iomanip>
namespace fs = std::filesystem;


struct LeafFeatures {
    double circularity;
    double elongation;
    double solidity;
    double hu[7];
};

struct LeafEntry {
    string speciesName;
    LeafFeatures features;
};

struct LeafTemplate {
    string speciesName;
    double area;
    double elongation;
    double circularity;
    double hu[7];
};

Mat resize_image(Mat src) {

    if (src.empty()) return src;
    int width = 500;
    double aspectRatio = (double)src.rows / src.cols;
    int height = (int)(width * aspectRatio);
    Mat resized;
    resize(src, resized, Size(256, 256), 0, 0, INTER_AREA);
	return resized;
}



// 1. Preprocessing (Segmentation & Binarization)
Mat segmentLeaf(Mat src) {
    Mat gray;
    /*if (src.channels() > 1) cvtColor(src, gray, COLOR_BGR2GRAY);
    else gray = src.clone();*/


    //Mat blurred;
    //GaussianBlur(gray, blurred, Size(5, 5), 0);

    int* hist = ex3_1_compute_hist(gray);
    float* pdf = ex3_2_compute_pdf(hist, gray.rows * gray.cols);


    int wh = 5;       
    float th = 0.00001f; 


    grayscale_mapping mapping = ex3_5_1_multi_level_thresholding(wh, th, pdf);


    Mat dst = ex3_5_3_draw_multiple_thresholding(gray, mapping);

    //Mat blurred;
    //GaussianBlur(gray, blurred, Size(5, 5), 0);


    //Mat clean = ex7_1_3_opening(binary);
    //clean = ex7_1_4_closing(clean);

    return dst;
}
Mat segmentByRemovingPink(Mat src) {
    Mat hsv, pinkMask;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // Pink/Magenta/Red Hue ranges from ~140 to 180
    Scalar lower_pink(140, 50, 50);
    Scalar upper_pink(180, 255, 255);

    // 1. Find the pink background
    inRange(hsv, lower_pink, upper_pink, pinkMask);

    // 2. Invert the mask
    Mat leafMask;
    bitwise_not(pinkMask, leafMask);

    // 3. Clean up noise
    Mat clean = ex7_1_3_opening(leafMask);
    clean = ex7_1_4_closing(clean);

    return clean;
}
Mat draw_contour1(contour cnt, Mat src) {
    if (src.empty()) {
        printf("Error: Input image for draw_contour1 is empty!\n");
        return Mat();
    }

    Mat dst;
    dst = src.clone(); // It's already color, just copy it

    // Now draw your points
    for (size_t i = 0; i < cnt.border.size(); i++) {
        Point1 p = cnt.border[i];
        // Note: Using Scalar(0, 0, 255) to make the circles RED
        circle(dst, Point(p.y, p.x), 1, Scalar(0, 0, 255), -1);
    }

    return dst;
}

contour extractLeafContour(Mat binary) {
    Point1 p0 = findPo(binary); 
    return ex6_1_1_trace_contour(binary, p0);
}

// 3. Descriptor Calculation
LeafFeatures calculateAll_orig(Mat binary, contour leafContour) {
    LeafFeatures f;

    // 1. Area & Perimeter
    double area = (double)ex4_1_2_compute_area(binary);

    // For perimeter, we use the size of the border vector from your contour struct
    double perimeter = (double)leafContour.border.size();

    // 2. Circularity Calculation
    f.circularity = (4 * CV_PI * area) / (perimeter * perimeter);

    // 3. Elongation
    f.elongation = ex4_1_4_compute_axis_of_elongation(binary);

    // 4. Solidity (Area / Convex Hull Area)
    vector<Point> cvPoints;
    for (auto p : leafContour.border) cvPoints.push_back(Point(p.y, p.x));

    vector<Point> hull;
    convexHull(cvPoints, hull);
    f.solidity = area / contourArea(hull);

    if (binary.channels() > 1) {
        cvtColor(binary, binary, COLOR_BGR2GRAY);
    }

    // 5. Hu Moments 
    Moments m = moments(binary, true);
    double huValues[7];
    HuMoments(m, huValues);
    for (int i = 0; i < 7; i++) f.hu[i] = huValues[i];

    return f;
}

LeafFeatures calculateAll(Mat binary, contour leafContour) {
    LeafFeatures f;

    vector<Point> cvPoints;
    for (auto p : leafContour.border) {
        cvPoints.push_back(Point(p.y, p.x));
    }

    double area = contourArea(cvPoints);
    if (area == 0) area = (double)ex4_1_2_compute_area(binary);

    double perimeter = arcLength(cvPoints, true);

    if (perimeter > 0) {
        f.circularity = (4 * CV_PI * area) / (perimeter * perimeter);
    }
    else {
        f.circularity = 0;
    }

    f.elongation = abs(ex4_1_4_compute_axis_of_elongation(binary));

    vector<Point> hull;
    if (cvPoints.size() > 0) {
        convexHull(cvPoints, hull);
        double hullArea = contourArea(hull);
        if (hullArea > 0) {
            f.solidity = area / hullArea;
        }
        else {
            f.solidity = 0;
        }
    }

    // 6. Hu Moments
    Mat gray = binary;
    if (binary.channels() > 1) {
        cvtColor(binary, gray, COLOR_BGR2GRAY);
    }

    Moments m = moments(gray, true);
    double huValues[7];
    HuMoments(m, huValues);
    for (int i = 0; i < 7; i++) f.hu[i] = huValues[i];

    return f;
}

double calculateDistance(LeafFeatures unknown, LeafTemplate target) {
    // We compare Elongation and Circularity 
    double d_elong = pow(unknown.elongation - target.elongation, 2);
    double d_circ = pow(unknown.circularity - target.circularity, 2);

    double d_hu = pow(unknown.hu[0] - target.hu[0], 2);

    return sqrt(d_elong + d_circ + d_hu);
}

string classifyLeaf(LeafFeatures unknown, vector<LeafTemplate>& database) {
    double minDistance = 1000000.0;
    string bestMatch = "Unknown";

    for (const auto& leafType : database) {
        double d = calculateDistance(unknown, leafType);

        if (d < minDistance) {
            minDistance = d;
            bestMatch = leafType.speciesName;
        }
    }
    return bestMatch;
}


void saveLeafToCSV(string species, LeafFeatures f) {
    // Open the file in "append" mode
    std::ofstream file("leaf_results.csv", std::ios::app);

    // If file is empty, write header first
    static bool headerWritten = false;
    if (!headerWritten) {
        file << "Species,Elongation,Circularity,Solidity,Hu1\n";
        headerWritten = true;
    }

    // Write the leaf data
    file << species << ","
        << f.elongation << ","
        << f.circularity << ","
        << f.solidity << ","
        << f.hu[0] << "\n";

    file.close();
}

void runSingleTest() {
    char fname[MAX_PATH];

    while (openFileDlg(fname)) 
        {
            Mat src = imread(fname);
            if (src.empty()) return;

            Mat inverted;
            cv::bitwise_not(src, inverted);

			Mat resized_image = resize_image(inverted);
			imshow("Original image", resized_image);

			contour extracted = extractLeafContour(resized_image);

			//imshow("Contour", ex6_1_contourTracing());


            waitKey(0);
        }
}


int main()
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
			printf(" 0 - Exit\n\n");
		}
		printf("Option: ");
		scanf("%d", &op);
		
        char fname[MAX_PATH];
        while (openFileDlg(fname)) {
            Mat src = imread(fname, IMREAD_GRAYSCALE);
            if (src.empty()) continue;

            Mat inverted;
            cv::bitwise_not(src, inverted);

			Mat resized_image = resize_image(inverted);


            threshold(resized_image, resized_image, 127, 255, THRESH_BINARY);

            Point1 p0 = findPo(resized_image);

            if (p0.x != -1) {

                contour cnt = ex6_1_1_trace_contour(resized_image, p0);


                Mat dst = draw_contour(cnt, resized_image);

                imshow("Contur Detectat", dst);
                printf("Contur gasit! Numar puncte: %zu\n", cnt.border.size());

                LeafFeatures feat = calculateAll(resized_image, cnt);
                // Print results
                cout << "--- LEAF RESULTS ---" << endl;
                cout << "Circularity: " << feat.circularity << endl;
                cout << "Elongation: " << feat.elongation << endl;
                cout << "Solidity: " << feat.solidity << endl;
                cout << "Hu[0]: " << feat.hu[0] << endl;
				cout << "Hu[1]: " << feat.hu[1] << endl;
                cout << "Hu[2]: " << feat.hu[2] << endl;
                cout << "Hu[3]: " << feat.hu[3] << endl;
                cout << "Hu[4]: " << feat.hu[4] << endl;
                cout << "Hu[5]: " << feat.hu[5] << endl;
                cout << "Hu[6]: " << feat.hu[6] << endl;
            }
            else {
                printf("Nu a fost gasit niciun obiect.\n");
            }

			

            waitKey();
        }
		
	} while (op != 0);
	return 0;
}


int main_cuEroare() {
    // 1. Define your species
    vector<string> species = { "1. Quercus suber", "2. Salix atrocinerea", "3. Populus nigra", "4. Alnus sp", "5. Quercus robur" };
    string rootPath = "D:/FACULTATE/SEM II/PI/PROIECT/leaf_dataset/BW/"; 

    cout << "Starting Leaf Feature Extraction..." << endl;
    cout << "Species, Elongation, Circularity, Solidity" << endl;

    // 2. Loop through each species
    for (const string& s : species) {
        string folderPath = rootPath + s;
        int count = 0;

        // 3. Loop through images in folder (limit to 10)
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            cout << "Absolute Path: " << fs::absolute(folderPath) << endl;
            if (count >= 10) break;

            // Skip directories if there are subfolders
            if (entry.is_directory()) continue;

            Mat src = imread(entry.path().string(), IMREAD_COLOR);
            if (src.empty()) continue;

			imshow("Original Image", resize_image(src));

            Mat inverted;
            cv::bitwise_not(src, inverted);

            Mat resized_image = resize_image(inverted);

            threshold(resized_image, resized_image, 127, 255, THRESH_BINARY);

            Point1 p0 = findPo(resized_image);

            if (p0.x != -1) {

                contour cnt = ex6_1_1_trace_contour(resized_image, p0);


                Mat dst = draw_contour(cnt, resized_image);


                LeafFeatures feat = calculateAll(resized_image, cnt);
                // 4. Save/Print for your Plot
                //cout << s << ", " << feat.elongation << ", " << feat.circularity << ", " << feat.solidity << endl;
                //saveLeafToCSV(s, feat);

            }
            else {
                printf("Nu a fost gasit niciun obiect.\n");
            }

            

            count++;
        }
    }

    cout << "Extraction Complete. Check leaf_results.csv for your plot data!" << endl;
    return 0;
}

int main_calculareCSV() {
    // 1. Dataset Configuration
    vector<string> species = {
        "1. Quercus suber",
        "2. Salix atrocinerea",
        "3. Populus nigra",
        "4. Alnus sp",
        "5. Quercus robur"
    };

    string rootPath = "D:/FACULTATE/SEM II/PI/PROIECT/leaf_dataset/BW/";

    cout << "Starting Leaf Feature Extraction..." << endl;
    cout << "Species, Elongation, Circularity, Solidity" << endl;

    for (const string& s : species) {
        string folderPath = rootPath + s;
        int count = 0;

        if (!fs::exists(folderPath)) {
            cout << "Folder not found: " << folderPath << endl;
            continue;
        }

        // 2. Iterate through the 20 images per folder
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (count >= 20) break;

            string imagePath = entry.path().string();
            Mat src = imread(imagePath, IMREAD_COLOR);
            if (src.empty()) continue;

            Mat inverted;
            cv::bitwise_not(src, inverted);
            Mat resized_image = resize_image(inverted);
            threshold(resized_image, resized_image, 127, 255, THRESH_BINARY);

            // --- STEP 4: CONTOUR TRACING (Your Tracer) ---
            Point1 p0 = findPo(resized_image);
            if (p0.x != -1) {
                contour cnt = ex6_1_1_trace_contour(resized_image, p0);

                // --- STEP 5: FEATURE EXTRACTION (Your Descriptors) ---
                if (cnt.border.size() > 20) {
                    // This function calls your area/elongation/circularity/hu functions
                    LeafFeatures feat = calculateAll(resized_image, cnt);

                    // Output results for the plot
                    cout << s << ", " << feat.elongation << ", " << feat.circularity
                        << ", " << feat.solidity << ", " << feat.hu[0] << endl;

                    // Save to your CSV function
                    saveLeafToCSV(s, feat);
                    count++;
                }
            }
        }
    }

    cout << "Extraction Complete. Use the CSV file to create your graph." << endl;
    return 0;
}

vector<LeafEntry> readCSV(string filename) {
    vector<LeafEntry> allEntries;
    ifstream file(filename);
    string line;

    // Sărim peste header (Species, Elongation, Circularity, Solidity, Hu1...)
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        LeafEntry entry;

        // 1. Numele speciei
        getline(ss, entry.speciesName, ',');

        // 2. Elongation
        getline(ss, val, ','); entry.features.elongation = stod(val);

        // 3. Circularity
        getline(ss, val, ','); entry.features.circularity = stod(val);

        // 4. Solidity
        getline(ss, val, ','); entry.features.solidity = stod(val);

        // 5. Momentele Hu (citim restul coloanelor)
        for (int i = 0; i < 7; i++) {
            if (getline(ss, val, ',')) {
                entry.features.hu[i] = stod(val);
            }
        }
        allEntries.push_back(entry);
    }
    return allEntries;
}

void drawGraph(vector<LeafEntry> data) {
    stringstream ss;
    ss << fixed << setprecision(2);

    int width = 900, height = 700, margin = 80;
    Mat canvas = Mat::zeros(height, width, CV_8UC3);
    canvas.setTo(Scalar(255, 255, 255));

    // 1. Găsim MIN și MAX folosind abs() pentru a evita valorile negative
    double minE = abs(data[0].features.elongation), maxE = abs(data[0].features.elongation);
    double minC = abs(data[0].features.circularity), maxC = abs(data[0].features.circularity);

    for (auto& entry : data) {
        double e = abs(entry.features.elongation);
        double c = abs(entry.features.circularity);
        if (e < minE) minE = e; if (e > maxE) maxE = e;
        if (c < minC) minC = c; if (c > maxC) maxC = c;
    }

    double rangeE = maxE - minE; if (rangeE == 0) rangeE = 1.0;
    double rangeC = maxC - minC; if (rangeC == 0) rangeC = 1.0;

    // 2. Desenăm axele
    line(canvas, Point(margin, height - margin), Point(width - margin, height - margin), Scalar(0, 0, 0), 2);
    line(canvas, Point(margin, height - margin), Point(margin, margin), Scalar(0, 0, 0), 2);

    // 1. VALORI PE AXA X(Elongation)
    // Valoarea minimă la începutul axei
    ss.str(""); ss << minE;
    putText(canvas, ss.str(), Point(margin - 10, height - margin + 20), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);

    // Valoarea maximă la sfârșitul axei
    ss.str(""); ss << maxE;
    putText(canvas, ss.str(), Point(width - margin - 40, height - margin + 20), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);

    // 2. VALORI PE AXA Y (Circularity)
    // Valoarea minimă (jos)
    ss.str(""); ss << minC;
    putText(canvas, ss.str(), Point(5, height - margin), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);

    // Valoarea maximă (sus)
    ss.str(""); ss << maxC;
    putText(canvas, ss.str(), Point(5, margin + 10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);

    // 3. OPȚIONAL: Grila sau gradații (mici linii pe axe)
    for (int i = 0; i <= 4; i++) {
        // Gradații pe X
        int tickX = margin + i * (width - 250) / 4;
        line(canvas, Point(tickX, height - margin), Point(tickX, height - margin + 5), Scalar(0, 0, 0), 1);

        // Gradații pe Y
        int tickY = (height - margin) - i * (height - 2 * margin) / 4;
        line(canvas, Point(margin - 5, tickY), Point(margin, tickY), Scalar(0, 0, 0), 1);
    }

    // 3. Desenăm punctele cu scalare forțată
    for (auto& entry : data) {
        double valE = abs(entry.features.elongation);
        double valC = abs(entry.features.circularity);

        // Mapare forțată pe pixeli
        int x = margin + (int)((valE - minE) / rangeE * (width - 250)); // lăsăm loc de legendă
        int y = (height - margin) - (int)((valC - minC) / rangeC * (height - 2 * margin));

        Scalar color = Scalar(128, 128, 128);
        if (entry.speciesName.find("1.") != string::npos) color = Scalar(255, 0, 0);
        else if (entry.speciesName.find("2.") != string::npos) color = Scalar(0, 180, 0);
        else if (entry.speciesName.find("3.") != string::npos) color = Scalar(0, 0, 255);
        else if (entry.speciesName.find("4.") != string::npos) color = Scalar(0, 150, 255);
        else if (entry.speciesName.find("5.") != string::npos) color = Scalar(255, 0, 255);

        circle(canvas, Point(x, y), 6, color, -1);
        circle(canvas, Point(x, y), 6, Scalar(0, 0, 0), 1);
    }

    // Etichete
    putText(canvas, "Elongation (X)", Point(width / 2 - 50, height - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(canvas, "Circularity (Y)", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

	putText(canvas, "1. Quercus suber", Point(width - margin - 150, margin + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
	putText(canvas, "2. Quercus robur", Point(width - margin - 150, margin + 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 180, 0), 1);
	putText(canvas, "3. Fagus sylvatica", Point(width - margin - 150, margin + 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
	putText(canvas, "4. Acer platanoides", Point(width - margin - 150, margin + 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 150, 255), 1);
	putText(canvas, "5. Betula pendula", Point(width - margin - 150, margin + 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    imshow("Feature Space - Leaf Classification", canvas);
    waitKey(0);
}

void drawGraph2(vector<LeafEntry> data) {
    if (data.empty()) return;

    int width = 900;
    int height = 700;
    int margin = 80;
    Mat canvas = Mat::zeros(height, width, CV_8UC3);
    canvas.setTo(Scalar(255, 255, 255));

    // 1. Găsim MIN și MAX 
    double minE = data[0].features.elongation, maxE = data[0].features.elongation;
    double minC = data[0].features.circularity, maxC = data[0].features.circularity;

    for (auto& entry : data) {
        if (entry.features.elongation < minE) minE = entry.features.elongation;
        if (entry.features.elongation > maxE) maxE = entry.features.elongation;
        if (entry.features.circularity < minC) minC = entry.features.circularity;
        if (entry.features.circularity > maxC) maxC = entry.features.circularity;
    }

    // Marjă de siguranță 
    double rangeE = (maxE - minE == 0) ? 1.0 : (maxE - minE) * 1.2;
    double rangeC = (maxC - minC == 0) ? 1.0 : (maxC - minC) * 1.2;

    // 2. Desenăm axele
    line(canvas, Point(margin, height - margin), Point(width - margin, height - margin), Scalar(0, 0, 0), 2);
    line(canvas, Point(margin, height - margin), Point(margin, margin), Scalar(0, 0, 0), 2);

    // 3. Desenăm punctele
    for (auto& entry : data) {
        // Mapare dinamică: (valoare - min) / range * spațiu_disponibil
        int x = margin + (int)((entry.features.elongation - minE) / rangeE * (width - 2 * margin));
        int y = (height - margin) - (int)((entry.features.circularity - minC) / rangeC * (height - 2 * margin));

        Scalar color;
        char firstChar = entry.speciesName[0];
        switch (firstChar) {
        case '1': color = Scalar(255, 0, 0); break;   // Albastru
        case '2': color = Scalar(0, 180, 0); break;   // Verde
        case '3': color = Scalar(0, 0, 255); break;   // Rosu
        case '4': color = Scalar(0, 150, 255); break; // Portocaliu
        case '5': color = Scalar(255, 0, 255); break; // Magenta
        default:  color = Scalar(100, 100, 100);
        }

        circle(canvas, Point(x, y), 6, color, -1);
        circle(canvas, Point(x, y), 6, Scalar(0, 0, 0), 1);
    }

    // 4. Etichete și Legendă (poziționată fix)
    putText(canvas, "Elongation (X)", Point(width / 2 - 50, height - 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    putText(canvas, "Circularity (Y)", Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);

    int legX = width - 200;
    putText(canvas, "1. Quercus suber", Point(legX, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
    putText(canvas, "2. Quercus robur", Point(legX, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 180, 0), 1);
    putText(canvas, "3. Fagus sylvatica", Point(legX, 140), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
    putText(canvas, "4. Acer platanoides", Point(legX, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 150, 255), 1);
    putText(canvas, "5. Betula pendula", Point(legX, 180), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    imshow("Feature Space - Leaf Classification", canvas);
    waitKey(0);
}

int main_grafic() {

    cout << "Generare grafic..." << endl;
    vector<LeafEntry> data = readCSV("D:/FACULTATE/SEM II/PI/PROIECT/OpenCVApplication-VS2022_OCV490_basic/leaf_results.csv");

    if (!data.empty()) {
        drawGraph(data);
    }
    else {
        cout << "Fișierul CSV este gol sau nu a fost găsit!" << endl;
    }

    return 0;
}