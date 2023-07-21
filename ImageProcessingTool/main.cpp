#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>

// Function to display an image with a given window title
void displayImage(const cv::Mat& image, const std::string& windowTitle) {
    cv::imshow(windowTitle, image);
    cv::waitKey(0);
}

// Function for image filtering (Step 4)
cv::Mat applyFilter(const cv::Mat& image, int kernelSize) {
    cv::Mat filteredImage;
    cv::GaussianBlur(image, filteredImage, cv::Size(kernelSize, kernelSize), 0);
    return filteredImage;
}

// Function for image transformation (Step 5)
cv::Mat applyTransformation(const cv::Mat& image, double scale, double rotationAngle, bool flipHorizontal) {
    cv::Mat transformedImage = image.clone();
    cv::resize(transformedImage, transformedImage, cv::Size(), scale, scale);
    cv::Point2f center(transformedImage.cols / 2.0, transformedImage.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotationAngle, 1.0);
    cv::warpAffine(transformedImage, transformedImage, rotationMatrix, transformedImage.size());
    if (flipHorizontal)
        cv::flip(transformedImage, transformedImage, 1);
    return transformedImage;
}

// Function for basic image editing tools (Step 6)
cv::Mat applyBasicEditing(const cv::Mat& image, int brightness, double contrast, double saturation) {
    cv::Mat editedImage = image.clone();
    editedImage.convertTo(editedImage, -1, contrast, brightness);
    if (saturation != 1.0) {
        cv::Mat hsvImage;
        cv::cvtColor(editedImage, hsvImage, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        cv::split(hsvImage, channels);
        channels[1] *= saturation;
        cv::merge(channels, hsvImage);
        cv::cvtColor(hsvImage, editedImage, cv::COLOR_HSV2BGR);
    }
    return editedImage;
}

// Function to concatenate images vertically
cv::Mat concatImagesVertically(const std::vector<cv::Mat>& images) {
    int totalHeight = 0;
    int maxWidth = 0;

    for (const auto& img : images) {
        totalHeight += img.rows;
        maxWidth = std::max(maxWidth, img.cols);
    }

    cv::Mat collage(totalHeight, maxWidth, images[0].type(), cv::Scalar::all(0));

    int y_offset = 0;
    for (const auto& img : images) {
        cv::Mat roi(collage, cv::Rect(0, y_offset, img.cols, img.rows));
        img.copyTo(roi);
        y_offset += img.rows;
    }

    return collage;
}

// Function for additional image processing operations (Step 7)
cv::Mat applyAdditionalProcessing(const cv::Mat& image, int edgeThreshold, int thresholdValue) {
    cv::Mat processedImage, edgesImage, thresholdedImage, contourImage;
    cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
    cv::Canny(processedImage, edgesImage, edgeThreshold, edgeThreshold * 3);
    cv::threshold(processedImage, thresholdedImage, thresholdValue, 255, cv::THRESH_BINARY);

    // Find contours on the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholdedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours on a 3-channel image
    cv::cvtColor(thresholdedImage, contourImage, cv::COLOR_GRAY2BGR);
    cv::drawContours(contourImage, contours, -1, cv::Scalar(255, 255, 255), 1);

    // Resize contourImage to match the height of processedImage
    cv::resize(contourImage, contourImage, processedImage.size());

    // Convert the edge image to a 3-channel image
    cv::cvtColor(edgesImage, edgesImage, cv::COLOR_GRAY2BGR);

    // Create the collage by concatenating the images vertically
    std::vector<cv::Mat> imagesToConcat = { processedImage, edgesImage, contourImage };
    cv::Mat collage = concatImagesVertically(imagesToConcat);

    return collage;
}

// Function for image blending and masking (Step 8)
cv::Mat applyBlendingAndMasking(const cv::Mat& image1, const cv::Mat& image2, double alpha) {
    // Ensure both images have the same size and data type
    cv::Mat image1_resized, image2_resized;
    cv::resize(image2, image2_resized, image1.size());
    image2_resized.convertTo(image2_resized, image1.type());

    // Create the blended image using addWeighted function
    cv::Mat blendedImage;
    cv::addWeighted(image1, alpha, image2_resized, 1.0 - alpha, 0, blendedImage);

    return blendedImage;
}

// Function for image feature detection (Step 9)
cv::Mat applyFeatureDetection(const cv::Mat& image) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat featureImage;
    sift->detect(image, keypoints);
    cv::drawKeypoints(image, keypoints, featureImage);
    return featureImage;
}

// Function for object detection (Step 10)
cv::Mat applyObjectDetection(const cv::Mat& image) {
    // Load the Haar Cascade Classifier for face detection
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("D:/PProject/ImageProcessingTool/haarcascade_frontalface_default.xml")) {
        std::cout << "Error: Could not load the cascade classifier!" << std::endl;
        return image;
    }

    // Convert image to grayscale for face detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Detect faces in the image
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.3, 5);

    // Draw rectangles around the detected faces
    cv::Mat resultImage = image.clone();
    for (const auto& face : faces) {
        cv::rectangle(resultImage, face, cv::Scalar(255, 0, 0), 2); // Draw a blue rectangle around each face
    }

    return resultImage;
}

// Function for image segmentation (Step 11)
cv::Mat applyImageSegmentation(const cv::Mat& image, int numClusters) {
    cv::Mat reshapedImage = image.reshape(1, image.cols * image.rows);
    reshapedImage.convertTo(reshapedImage, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(reshapedImage, numClusters, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
        3, cv::KMEANS_RANDOM_CENTERS, centers);

    cv::Mat segmented_image(image.size(), image.type());
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int label = labels.at<int>(i * image.cols + j);
            segmented_image.at<cv::Vec3b>(i, j) = centers.at<cv::Vec3f>(label);
        }
    }

    return segmented_image;
}

// Function for image registration using SIFT (Step 12)
cv::Mat applyImageRegistration(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();

    // Detect keypoints and extract descriptors for both images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Match keypoints between the two images using a matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Filter out good matches based on a distance threshold
    double maxDist = 0.1;
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i].distance < maxDist) {
            goodMatches.push_back(matches[i]);
        }
    }

    // Draw the matches on a new image
    cv::Mat matchedImage;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, matchedImage);

    return matchedImage;
}

// Function to save the processed image (Step 13)
bool saveProcessedImage(const cv::Mat& processedImage, const std::string& outputFilePath) {
    if (processedImage.empty()) {
        std::cout << "Error: Processed image is empty." << std::endl;
        return false;
    }

    bool success = cv::imwrite(outputFilePath, processedImage);
    if (success) {
        std::cout << "Processed image has been saved to: " << outputFilePath << std::endl;
        return true;
    }
    else {
        std::cout << "Error: Failed to save the processed image." << std::endl;
        return false;
    }
}



int main() {

    cv::Mat image1 = cv::imread("D:/PProject/ImageProcessingTool/ImageProcessingTool/game.jpg");
    cv::Mat image2 = cv::imread("D:/Resume/passport.jpg");

    // Check if the images were loaded successfully
    if (image1.empty() || image2.empty()) {
        std::cout << "Error: Could not read the images!" << std::endl;
        return -1;
    }

    // Perform image filtering (Step 4)
    cv::Mat filteredImage = applyFilter(image1, 5);

    // Perform image transformation (Step 5)
    cv::Mat transformedImage = applyTransformation(image2, 0.7, 30.0, true);

    // Perform basic image editing tools (Step 6)
    cv::Mat editedImage = applyBasicEditing(image1, 50, 1.2, 0.8);

    // Perform additional image processing operations (Step 7)
    cv::Mat processedImage = applyAdditionalProcessing(image2, 50, 100);

    // Perform image blending and masking (Step 8)
    cv::Mat blendedImage = applyBlendingAndMasking(image1, image2, 0.5);

    // Perform image feature detection (Step 9)
    cv::Mat featureImage = applyFeatureDetection(image1);

    // Perform object detection (Step 10)
    cv::Mat detectedImage = applyObjectDetection(image2);

    // Perform image segmentation (Step 11)
    cv::Mat segmentedImage = applyImageSegmentation(image1, 4);

    // Perform image registration using SIFT (Step 12)
    cv::Mat registeredImage = applyImageRegistration(image1, image2);

    // Step 2: Save the processed image to a file
    std::string outputFilePath = "D:/PProject/ImageProcessingTool/ImageProcessingTool/images/processed_image1.jpg";
    saveProcessedImage(processedImage, outputFilePath);

    cv::imshow("Original Image 1", image1);
    cv::imshow("Original Image 2", image2);
    cv::imshow("Filtered Image", filteredImage);
    cv::imshow("Transformed Image", transformedImage);
    cv::imshow("Edited Image", editedImage);
    cv::imshow("Processed Image", processedImage);
    cv::imshow("Blended Image", blendedImage);
    cv::imshow("Feature Image", featureImage);
    cv::imshow("Detected Image", detectedImage);
    cv::imshow("Segmented Image", segmentedImage);
    cv::imshow("Registered Image", registeredImage);

    cv::waitKey(0);

    return 0;
}
