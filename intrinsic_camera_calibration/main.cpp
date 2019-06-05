#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

// real time intrinsic camera calibration

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>



using namespace std;
using namespace cv;

const float square_dimension = 0.03f; //meters
const Size chessboard_corners = Size(6, 9);

//chessboard corners extraction

void create_board_position(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

//getting the corners
void get_chessboard_corners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}
		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("Looking for Corners", *iter);
			waitKey(0);
		}
	}
}

//calibrate intrinsic camera
void camera_calibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distortionCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	get_chessboard_corners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	create_board_position(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distortionCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distortionCoefficients, rVectors, tVectors);

}


//save calibration in a text file
bool save_calibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}

		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}

		}

		outStream.close();
		return true;
	}

	return false;
}


int main(int , char** )
{
	// create mat object for video frame and drawing corners detection to frame
	Mat frame;
	Mat drawToFrame;

	// object to store calibration result
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	// capture the video
	VideoCapture vid;

	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // autodetect default API

	// open selected camera using selected API
	vid.open(deviceID + apiID);

	//check for opening
	if (!vid.isOpened())
	{
		return -1;
	}

	//initialize frame per second
	int fps = 20;

	namedWindow("mounted camera", WINDOW_AUTOSIZE);

	//Real time calibration
	while (vid.read(frame))
	{
		if (frame.empty()) 
		{
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboard_corners, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		
		frame.copySize(drawToFrame);

		//drawing from corners
		drawChessboardCorners(drawToFrame, chessboard_corners, foundPoints, found);

		if (found)
			imshow("mounted camera", drawToFrame);
		else
			imshow("mounted camera", frame);

		//timing for waitkey
		char character = waitKey(100000/fps);


		switch (character)
		{
		case ' ':
			//saving image
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);

			}
			break;
		case 13:
			//start calibration
			if (savedImages.size() > 15)
			{
				camera_calibration(savedImages, chessboard_corners, square_dimension, cameraMatrix, distanceCoefficients);
				save_calibration("intrinsic_calibration.txt", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//exit
			return 0;
			break;
		}

		if (waitKey(5) >= 0)
			break;
		
	}

	return 0;
}