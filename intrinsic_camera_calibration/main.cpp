/* Program to perform real time camera calibration
 * input is given in the example at calibration images file
 * it is chessboard with 9 x 6 corners and 0.03 meters dimension [PLEASE PRINT IT!]
 * the output are intrinsic camera matrix, distortion coefficients and reprojection error
 * the accuracy of detected corners are sub pixels
*/

#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

// C++ HEADER NEEDED TO PERFORM REAL TIME CAMERA CALIBRATION
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <iomanip>
#include <vector>

// OPENCV HEADER NEEDED TO PERFORM REAL TIME CAMERA CALIBRATION
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>


using namespace cv;
using namespace std;


// function to creates world cooridnates of chessboard, Z axis is approximated as 0
void world_coordinates(Size board_size, float chess_dimension, vector<Point3f>& worldPoints)
{
	for (int i = 0; i < board_size.height; i++)
	{
		for (int j = 0; j < board_size.width; j++)
		{
			worldPoints.push_back(Point3f(j*chess_dimension, i*chess_dimension, 0.0f));
		}

	}
}


// function to get image coordinate of corners
void get_image_coordinates(vector<Mat> images, vector<vector<Point2f>>& image_points, Size board_size)
{
	//cout << "\n" << images.size() << endl;
	for (int i = 0; i < images.size(); i++)
	{
		// vector to store corners
		vector<Point2f> store_points;

		// Finding corners location on the chessboard and store it in storePoints vector
		bool found = findChessboardCorners(images.at(i), board_size, store_points, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

		// getting more accurate pixel location of corners
		if (found)
		{
			// Mat object to store greyscale image of selected frame
			Mat convert_to_gray;

			// convert selected images to gray scale for input into subpixel corner detection function "cornerSubPix()"
			cvtColor(images.at(i), convert_to_gray, COLOR_BGR2GRAY);

			// function to get more accurate subpixel location of corners
			cornerSubPix(convert_to_gray, store_points, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

			image_points.push_back(store_points);
		}

	}

}


// function to calibrate camera
void camera_calibration(vector<Mat> images, Size board_size, float chess_dimension, ofstream& csvCameraMatrix, ofstream& csvDistCoeffs, ofstream& csvReprojErrors)
{
	// Mat object to store intrinsic camera parameters (camera matrix)
	Mat camera_matrix;

	// vector to store distortion matrix
	vector<float> dist_coeff_matrix;

	//vector for storing corners coordinates for every frame
	vector<vector<Point2f>> image_points;

	// obtain image coordinates of corners for every frame
	get_image_coordinates(images, image_points, board_size);

	vector<vector<Point3f>> world_points(1);

	// obtain world coordinates of chessboard for every frame
	world_coordinates(board_size, chess_dimension, world_points[0]);

	// making (resizing) the same world points remain for every corresponding Image points 
	world_points.resize(image_points.size(), world_points[0]);

	//vector for Rotation and Translation (Extrinsic Matrix) in each frame
	vector<Mat> rvectors, tvectors;

	// function computes camera matrix, extrinsic matrix and mean reprojection error corresponse to a set of images 
	// method used in this calibration is Zhang's Method 
	double error = calibrateCamera(world_points, image_points, board_size, camera_matrix, dist_coeff_matrix, rvectors, tvectors);

	// printing out the output
	cout << "\nIntrinsic camera matrix:  \n\n" << camera_matrix << endl;
	cout << "\nreprojection error:  \n\n" << error << endl;
	cout << "\ndistortion coefficients:  \n\n";

	// prints distortion coefficients (k1, k2, p1, p2[, k3[, k4, k5, k6]])
	for (int i = 0; i < dist_coeff_matrix.size(); i++)
	{
		cout << dist_coeff_matrix.at(i) << "   ";
	}

	// Mat object to store undistorted image
	Mat undistorted;

	// upgrade image points to compensate for lens distortion 
	undistort(images.at(0), undistorted, camera_matrix, dist_coeff_matrix);

	namedWindow("DISTORTED", WINDOW_AUTOSIZE);

	imshow("DISTORTED", images.at(0));

	namedWindow("UNDISTORTED", WINDOW_AUTOSIZE);

	imshow("UNDISTORTED", undistorted);

	// outputing intrinsic camera matrix result into text file (.txt)
	csvCameraMatrix << setprecision(8) << camera_matrix.at<double>(0, 0) << " " << camera_matrix.at<double>(0, 1) << " " << camera_matrix.at<double>(0, 2) << endl;
	csvCameraMatrix << setprecision(8) << camera_matrix.at<double>(1, 0) << " " << camera_matrix.at<double>(1, 1) << " " << camera_matrix.at<double>(1, 2) << endl;
	csvCameraMatrix << setprecision(8) << camera_matrix.at<double>(2, 0) << " " << camera_matrix.at<double>(2, 1) << " " << camera_matrix.at<double>(2, 2) << endl;

	// outputing distortion coefficients result into text file (.txt)
	for (int i = 0; i < dist_coeff_matrix.size(); i++)
	{
		csvDistCoeffs << dist_coeff_matrix.at(i) << "   ";
	}

	// outputing reprojection error result into text file (.txt)
	csvReprojErrors << error << endl;

	// optionally, we can store the result into yml file in opencv file storage class
	FileStorage storage("camera_calibration_results.yml", cv::FileStorage::WRITE);

	storage << "Intrinsic_camera_matrix" << camera_matrix;

	storage << "distortion_coefficients" << dist_coeff_matrix;

	storage << "reprojection_error" << error;

	storage.release();

	waitKey(0);

}


// function to perform real time camera calibration
int performing_calibration(Size board_size, float chess_dimension, int frames_number, ofstream& csvCameraMatrix, ofstream& csvDistCoeffs, ofstream& csvReprojErrors)
{
	int count = 0;

	Mat frame;

	// store frames selected for calibration
	vector<Mat> selected_image;

	// capturing video with default camera (0)
	VideoCapture vid(0);

	// disable autofocus in the camera
	vid.set(CAP_PROP_AUTOFOCUS, 0);

	// setting image width resolution in pixels
	vid.set(CAP_PROP_FRAME_WIDTH, 640);

	// setting image height resolution in pixels
	vid.set(CAP_PROP_FRAME_HEIGHT, 480);

	if (!vid.isOpened())
	{
		return 0;
	}

	// setting frame frequency (frame per second)
	int fps = 20;

	namedWindow("Webcam", WINDOW_AUTOSIZE);

	// keep recording frames until specified "frames_number" detected and selected for calibration are still unfulfilled

	while (count != frames_number)
	{
		// checking the frame reading
		if (!vid.read(frame))
		{
			break;
		}

		//vector to be filled by the detected corners
		vector<Point2f> corners;

		bool patternfound = findChessboardCorners(frame,
			board_size,
			corners,// finds the location of chessboard corners  and store it in "corners".
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);// CALIB_CB_FAST_CHECK is used to simplify the task and saves a lot of time on images have no chessboard corners

// Mat object to show paterned image
		Mat pattern_frame;

		// copying frame with patterned corners to pattern_frame Mat object
		frame.copyTo(pattern_frame);

		// function to draws detected chessboard corners 
		drawChessboardCorners(pattern_frame, board_size, Mat(corners), patternfound);

		// checking if chessboard corners pattern is detected 
		// show patterned image of frame if the pattern is found and show original frame if it is not found
		if (patternfound)
		{
			// shows patterned image
			imshow("Webcam", pattern_frame);

			// based on choice,if key "ENTER" is pressed, frame will be selected
			// any other key is taken as rejected!
			char ch = waitKey(0);

			if (ch == 13)
			{
				cout << count + 1;

				// storing frame in selected_image vector
				selected_image.push_back(frame.clone());

				count++;
			}

		}
		else
		{
			imshow("Webcam", frame);
		}

		// wait for specific amount of time before capturing subsequent frames
		waitKey(1000 / fps);
	}

	// closing webcam
	vid.release();

	// exterminate window for webcam
	destroyWindow("Webcam");

	// function to calibrate camera
	camera_calibration(selected_image, board_size, chess_dimension, csvCameraMatrix, csvDistCoeffs, csvReprojErrors);

	return 0;
}


int main(int argc, char** argv)
{
	// chessboard square dimension in meters
	float chess_dimension;

	cout << "\nEnter Chessboard squares dimesions in meteres (floating points format): ";

	cin >> chess_dimension;

	cout << endl;

	// width and height from chessboard pattern
	int width, height;

	cout << "\nEnter number of Chessboard's squares corners on its width: ";

	// from example of chessboard images in calibration_images file , width = 9
	cin >> width;

	cout << endl;

	cout << "\nEnter number of Chessboard's squares corners on its height: ";

	// from example of chessboard images in calibration_images file , height = 6
	cin >> height;

	cout << endl;

	int frames_number;

	// it is recomended to take more that 15 images to get more accurate results
	cout << "\nnumber of images for calibrating the camera";
	cout << "\n(taking large number of images will give better accuracy)";
	cout << "\nEnter number of images: ";

	cin >> frames_number;

	cout << "\n" << endl;

	// number of corners
	Size boardSize(width, height);

	// perform real time calibration

	ofstream csvCameraMatrix("../data/intrinsic_camera_matrix.txt");
	ofstream csvDistCoeffs("../data/distortion_coefficients");
	ofstream csvReprojErrors("../data/reprojection_errors");

	performing_calibration(boardSize, chess_dimension, frames_number, csvCameraMatrix, csvDistCoeffs, csvReprojErrors);

	csvCameraMatrix.close();
	csvDistCoeffs.close();
	csvReprojErrors.close();

	return 0;
}
