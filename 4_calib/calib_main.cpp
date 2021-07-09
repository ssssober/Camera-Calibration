#include "opencv2/opencv.hpp"
#include <vector>
#include <fstream>
#include <string>
using namespace std;
using namespace cv;

/*
*** 批量程序：求解从相机视野之外的世界坐标(Xw,Yw,Zw)到相机坐标系(Xc,Yc,Zc)之前的转换关系RT
*/

// 补偿镜子厚度和棋盘厚度
#define MIRROR_OFFSET 3
#define CHESSBOARD_OFFSET 5

// 镜子上贴棋盘格尺寸
#define CHESSBOARD_HEIGHT 11
#define CHESSBOARD_WIDTH  16
#define CHESSBOARD_SIZE   30

// 镜子中虚像棋盘格尺寸
#define CHESSBOARD_HEIGHT_SCREEN 7
#define CHESSBOARD_WIDTH_SCREEN  18
#define CHESSBOARD_SIZE_SCREEN   56.7

void CalculateTFM(Mat &tfm, Mat rvecs, Mat tvecs);

int main(int argc, char *argv[]) {
	string floder_path = "..\\20210528\\1_97";  //存放相机棋盘格图片的目录
	vector<cv::String> fileNames;
	fileNames.clear();
	glob(floder_path + "//*.jpg", fileNames, false);  // 棋盘格图片.jpg后缀
	size_t count = fileNames.size();
	for (int i = 0; i < count; i++){
		Mat ir_src = imread(fileNames[i], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		//cout << fileNames[i] << endl << endl;
		//
		string out_path = fileNames[i].substr(0, fileNames[i].length() - 4); //
		if (ir_src.empty()) {
			cout << "Images Loading Error!" << endl;
			return -1;
		}

		cvtColor(ir_src, ir_src, CV_BGR2GRAY);
		if (ir_src.depth() != CV_8U) {
			Mat ir_tmp = Mat::zeros(ir_src.size(), CV_8U);
			ir_src.convertTo(ir_tmp, CV_8U);
			ir_tmp.copyTo(ir_src);
		}

		// Load Camera Param
		string cameparamfile = floder_path + "//param-RGBD.yml";  //相机内参
		FileStorage fs(cameparamfile, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("\nCould not open the configuration file!");
			return -2;
		}
		cv::Mat cameraMatrix_left, distCoeffs_left;
		fs["CameraMatrix_Left"] >> cameraMatrix_left;
		fs["DistCoeffs_Left"] >> distCoeffs_left;

		float f_x = cameraMatrix_left.at<double>(0, 0);
		float f_y = cameraMatrix_left.at<double>(1, 1);
		float c_x = cameraMatrix_left.at<double>(0, 2);
		float c_y = cameraMatrix_left.at<double>(1, 2);

		Mat RT21 = Mat::eye(4, 4, CV_64FC1);
		Mat RT31 = Mat::eye(4, 4, CV_64FC1);

		// 计算RT21
		if (1) {
			bool found;
			vector<Point2f> pointBuf;
			Size boardSize = Size(CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT);

			found = findChessboardCorners(ir_src, boardSize, pointBuf,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
			if (found)                // If done with success,
			{
				// improve the found corners' coordinate accuracy for chessboard
				cornerSubPix(ir_src, pointBuf, Size(11, 11),
					Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

				Mat img_show = ir_src.clone();
				cvtColor(img_show, img_show, CV_GRAY2BGR);
				drawChessboardCorners(img_show, boardSize, pointBuf, found);
				//imshow("chessboard", img_show);
				//waitKey();
				string save_path1 = out_path;
				imwrite(save_path1.append(" _chessboard_2.png"), img_show);

				//------------------Calculate Camera Pose------------
				vector<Point3f> ObjectCorners;
				for (int i = 0; i < CHESSBOARD_HEIGHT; ++i)
				for (int j = 0; j < CHESSBOARD_WIDTH; ++j)
					ObjectCorners.push_back(Point3f(float(j*CHESSBOARD_SIZE), float(i*CHESSBOARD_SIZE), 0));

				Mat rvecs, tvecs;
				Mat cameraM = Mat::zeros(Size(3, 3), CV_64FC1);
				cameraM.at<double>(0, 0) = f_x;
				cameraM.at<double>(1, 1) = f_y;
				cameraM.at<double>(0, 2) = c_x;
				cameraM.at<double>(1, 2) = c_y;
				cameraM.at<double>(2, 2) = 1.0;
				Mat distCoe = Mat::zeros(Size(5, 1), CV_64FC1);
				solvePnPRansac(ObjectCorners, pointBuf, cameraM, distCoe, rvecs, tvecs);
				CalculateTFM(RT21, rvecs, tvecs);
			}
			else {
				cout << "Can not find Chessboard Corners!" << endl;
				return -1;
			}
		}

		// 计算RT31
		if (1) {
			bool found;
			vector<Point2f> pointBuf;
			Size boardSize = Size(CHESSBOARD_WIDTH_SCREEN, CHESSBOARD_HEIGHT_SCREEN);

			Mat ir_src_f;
			flip(ir_src, ir_src_f, 1);

			found = findChessboardCorners(ir_src_f, boardSize, pointBuf,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
			if (found)                // If done with success,
			{
				// improve the found corners' coordinate accuracy for chessboard
				cornerSubPix(ir_src_f, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
				Mat img_show = ir_src_f.clone();
				cvtColor(img_show, img_show, CV_GRAY2BGR);
				drawChessboardCorners(img_show, boardSize, pointBuf, found);
				//imshow("chessboard_flip", img_show);
				//waitKey();
				string save_path2 = out_path;
				imwrite(save_path2.append(" _chessboard_3_flip.png"), img_show);

				//------------------Calculate Camera Pose------------
				vector<Point3f> ObjectCorners;
				for (int i = 0; i < CHESSBOARD_HEIGHT_SCREEN; ++i)
				for (int j = 0; j < CHESSBOARD_WIDTH_SCREEN; ++j)
					ObjectCorners.push_back(Point3f(float(j*CHESSBOARD_SIZE_SCREEN), float(i*CHESSBOARD_SIZE_SCREEN), 0));

				Mat rvecs, tvecs;
				Mat cameraM = Mat::zeros(Size(3, 3), CV_64FC1);
				//内参
				cameraM.at<double>(0, 0) = f_x;
				cameraM.at<double>(1, 1) = f_y;
				cameraM.at<double>(0, 2) = c_x;
				cameraM.at<double>(1, 2) = c_y;
				cameraM.at<double>(2, 2) = 1.0;
				Mat distCoe = Mat::zeros(Size(5, 1), CV_64FC1); //畸变

				// flip检测角点后，坐标x进行转换到原始图片下
				for (int i = 0; i < pointBuf.size(); i++) {
					pointBuf[i].x = ir_src.cols - pointBuf[i].x;
				}

				Mat img_show_src = ir_src.clone();
				cvtColor(img_show_src, img_show_src, CV_GRAY2BGR);
				drawChessboardCorners(img_show_src, boardSize, pointBuf, found);
				//imshow("chessboard_src", img_show_src);
				//waitKey();
				string save_path3 = out_path;
				imwrite(save_path3.append("_chessboard_3_src.png"), img_show_src);
				solvePnPRansac(ObjectCorners, pointBuf, cameraM, distCoe, rvecs, tvecs);
				CalculateTFM(RT31, rvecs, tvecs);
			}
			else {
				cout << "Can not find Chessboard Corners!" << endl;
				return -1;
			}
		}

		Mat RT12 = RT21.inv();
		Mat RT32 = RT12 * RT31;
		Mat RT23 = RT32.inv();

		double pt0_3_[4] = { 0, 0, 0, 1 };
		double pt1_3_[4] = { 100, 0, 0, 1 };
		double pt2_3_[4] = { 0, 100, 0, 1 };
		Mat pt0_3 = Mat(4, 1, CV_64FC1, pt0_3_);
		Mat pt1_3 = Mat(4, 1, CV_64FC1, pt1_3_);
		Mat pt2_3 = Mat(4, 1, CV_64FC1, pt2_3_);

		Mat pt0_2 = RT32 * pt0_3;
		Mat pt1_2 = RT32 * pt1_3;
		Mat pt2_2 = RT32 * pt2_3;

		pt0_2.at<double>(2, 0) -= (MIRROR_OFFSET + CHESSBOARD_OFFSET);
		pt1_2.at<double>(2, 0) -= (MIRROR_OFFSET + CHESSBOARD_OFFSET);
		pt2_2.at<double>(2, 0) -= (MIRROR_OFFSET + CHESSBOARD_OFFSET);

		Mat x_3_ = pt1_2 - pt0_2;
		Mat y_3_ = pt2_2 - pt0_2;

		double x_3_c = sqrt(pow(x_3_.at<double>(0, 0), 2) + pow(x_3_.at<double>(1, 0), 2) + pow(x_3_.at<double>(2, 0), 2));
		double y_3_c = sqrt(pow(y_3_.at<double>(0, 0), 2) + pow(y_3_.at<double>(1, 0), 2) + pow(y_3_.at<double>(2, 0), 2));

		x_3_ /= x_3_c;
		y_3_ /= y_3_c;

		Mat pt0_4_ = pt0_2.clone();
		pt0_4_.at<double>(2, 0) *= -1;
		pt0_4_.at<double>(2, 0) += (MIRROR_OFFSET + CHESSBOARD_OFFSET);

		Mat pt1_4_ = pt1_2.clone();
		pt1_4_.at<double>(2, 0) *= -1;
		pt1_4_.at<double>(2, 0) += (MIRROR_OFFSET + CHESSBOARD_OFFSET);

		Mat pt2_4_ = pt2_2.clone();
		pt2_4_.at<double>(2, 0) *= -1;
		pt2_4_.at<double>(2, 0) += (MIRROR_OFFSET + CHESSBOARD_OFFSET);

		Mat x_4_ = pt1_4_ - pt0_4_;
		Mat y_4_ = pt2_4_ - pt0_4_;

		double x_4_c = sqrt(pow(x_4_.at<double>(0, 0), 2) + pow(x_4_.at<double>(1, 0), 2) + pow(x_4_.at<double>(2, 0), 2));
		double y_4_c = sqrt(pow(y_4_.at<double>(0, 0), 2) + pow(y_4_.at<double>(1, 0), 2) + pow(y_4_.at<double>(2, 0), 2));

		x_4_ /= x_4_c;
		y_4_ /= y_4_c;

		Mat x_4 = x_4_(Rect(0, 0, 1, 3)).clone();
		Mat y_4 = y_4_(Rect(0, 0, 1, 3)).clone();
		Mat z_4 = x_4.cross(y_4);

		x_4 = x_4.t();
		y_4 = y_4.t();
		z_4 = z_4.t();

		Mat R24 = Mat::eye(3, 3, CV_64FC1);
		x_4.copyTo(R24(Rect(0, 0, 3, 1)));
		y_4.copyTo(R24(Rect(0, 1, 3, 1)));
		z_4.copyTo(R24(Rect(0, 2, 3, 1)));

		Mat R42 = R24.inv();
		Mat RT42 = Mat::eye(4, 4, CV_64FC1);
		R42.copyTo(RT42(Rect(0, 0, 3, 3)));
		pt0_4_.copyTo(RT42(Rect(3, 0, 1, 4)));

		Mat RT24 = RT42.inv();
		Mat RT41 = RT21 * RT42;

		// 保存RT41结果到txt文档！
		// RT41：从相机视野之外的平面世界坐标到相机坐标系的转换RT
		string save_path4 = out_path;
		ofstream fout(save_path4.append("_RT41.txt"));
		fout << "RT41: \n";
		fout << RT41 << endl << endl;
		cout << " " << fileNames[i] << "  RT41参数保存完毕... ... ..." << endl << endl;
	}

	//system("pause");
	return 0;
}

void CalculateTFM(Mat &tfm, Mat rvecs, Mat tvecs)
{
	Mat Rotat = Mat::zeros(3, 3, CV_64F);
	Rodrigues(rvecs, Rotat);
	
	Mat tf_Mat = Mat::eye(4, 4, CV_64F);
	Rotat.copyTo(tf_Mat(Rect(0, 0, 3, 3)));
	tvecs.copyTo(tf_Mat(Rect(3, 0, 1, 3)));
	
	tf_Mat.copyTo(tfm);
}
