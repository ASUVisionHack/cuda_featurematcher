#include <iostream>
#include <iomanip>

#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace std;
using namespace cv;
using namespace cv::cuda;

struct Feature {
    vector<KeyPoint> keypoints;
    cv::Mat descriptors;
};

void localizeInImage(const std::vector<DMatch>& good_matches,
    const std::vector<KeyPoint>& keypoints_object,
    const std::vector<KeyPoint>& keypoints_scene, const Mat& img_object,
    const Mat& img_matches)
{
//-- Localize the object
std::vector<Point2f> obj;
std::vector<Point2f> scene;
for (int i = 0; i < good_matches.size(); i++) {
    //-- Get the keypoints from the good matches
    obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
    scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
}

try {
    Mat H = findHomography(obj, scene, RANSAC);
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(img_object.cols, 0);
    obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
    obj_corners[3] = cvPoint(0, img_object.rows);
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform(obj_corners, scene_corners, H);
    // Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0),
            scene_corners[1] + Point2f(img_object.cols, 0),
            Scalar(255, 0, 0), 4);
    line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0),
            scene_corners[2] + Point2f(img_object.cols, 0),
            Scalar(255, 0, 0), 4);
    line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0),
            scene_corners[3] + Point2f(img_object.cols, 0),
            Scalar(255, 0, 0), 4);
    line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0),
            scene_corners[0] + Point2f(img_object.cols, 0),
            Scalar(255, 0, 0), 4);
} catch (Exception& e) {}
}

void drawBoundingRect(const std::vector<DMatch> &good_matches,
    const std::vector<KeyPoint>& keypoints_object,
    const std::vector<KeyPoint>& keypoints_scene,
    const int object_col,
    Mat& img_matches) {

    vector<Point> image_points;
    for (int i=0; i<good_matches.size(); i++) {
        image_points.push_back(keypoints_scene[good_matches[i].trainIdx].pt + Point2f(object_col, 0));
    }

    Rect r = boundingRect(image_points);
    r.width += 0.6*r.width;
    r.height += 0.6*r.height;
    rectangle(img_matches, r, Scalar(255, 0, 0));

}

Feature processImageCPU(string inputFilename) {
    Mat object_img = imread(inputFilename, IMREAD_GRAYSCALE);

    if (!object_img.data) {
        cout << "error reading image" << inputFilename << endl;
    }

    cout << object_img.rows << " " << object_img.cols << endl;

    Feature feature;
    Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(50);
    surf->detectAndCompute(object_img, cv::Mat(), feature.keypoints, feature.descriptors);

    return feature;
}

void matchImageGPU(string inputFilename, Feature feature) {
    Mat image_scene = imread(inputFilename, IMREAD_GRAYSCALE);

    if (!image_scene.data) {
        cout << "error reading image" << inputFilename << endl;
    }

    GpuMat image_scene_GPU(image_scene);
    GpuMat descriptors_scene_GPU, keypoints_scene_GPU;

    // Get SURF vector for scene image
    cuda::SURF_CUDA surf(400);
    surf(image_scene_GPU, GpuMat(), keypoints_scene_GPU, descriptors_scene_GPU);

    // Match against known feature vector.
    GpuMat descriptors_object_GPU(feature.descriptors);
    cout << feature.descriptors.size() << endl;
    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher();
    vector< vector<DMatch> > matches;
    matcher->knnMatch(descriptors_object_GPU, descriptors_scene_GPU, matches, 2);

    // Download results from GPU
    vector<KeyPoint> keypoints_scene;
    surf.downloadKeypoints(keypoints_scene_GPU, keypoints_scene);

    Mat o = imread("29.jpeg", IMREAD_GRAYSCALE);

    std::vector< DMatch > good_matches;
    for (int k=0; k < min(feature.keypoints.size()-1, matches.size()); k++) {
        if ((matches[k][0].distance < 0.0*(matches[k][1].distance)) && 
            ((int)matches[k].size() <= 2 && (int)matches[k].size()>0)) {
                good_matches.push_back(matches[k][0]);
            }
    }

    Mat image_matches;
    drawMatches(o, feature.keypoints, image_scene, keypoints_scene, good_matches, image_matches);
    //localizeInImage(good_matches, feature.keypoints, keypoints_scene, o, image_matches);
    drawBoundingRect(good_matches, feature.keypoints, keypoints_scene, o.cols, image_matches);

    imshow("matches", image_matches);
    waitKey(0);




}

void processScene(string inputFilename, Feature ) {
    Mat img_object = imread(inputFilename, IMREAD_GRAYSCALE);


}

int main (int argc, const char *argv[]) {
    if (getCudaEnabledDeviceCount()) {
        cout << "CUDA is enabled" << endl;
    }
    Feature feature = processImageCPU("29.jpeg");
    matchImageGPU("5.jpeg", feature);
    // cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    return 0;

} 