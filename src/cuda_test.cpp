#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main (int argc, const char *argv[]) {
    cout << getCudaEnabledDeviceCount() << endl;
    cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    return 0;

} 