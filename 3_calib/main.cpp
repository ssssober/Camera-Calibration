#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <cv.h>   //cv.h OpenCV的主要功能头文件，务必要；
#include <highgui.h>
//C
#include <stdio.h>
using namespace std;
using namespace cv;

int read_reference_lines()
{
	int line = 0;

	ifstream fin("..\\reference.txt");
	if (!fin)
	{
		cout << "cannot open reference the file";
		return -1;
	}

	char c;
	while (fin.get(c))
	{
		if (c == '\n')
			line++;
	}

	fin.close();
	return line;
}

int read_deformed_lines()
{
	int line = 0;

	ifstream fin("..\\deformed.txt");
	if (!fin)
	{
		cout << "cannot open reference the file";
		return -1;
	}

	char c;
	while (fin.get(c))
	{
		if (c == '\n')
			line++;
	}

	fin.close();
	return line;
}

bool read_reference(CvMat* reference_x, CvMat* reference_y, CvMat* reference_z, int total_lines)
{
	ifstream fin("..\\reference.txt");
	if (!fin)
	{
		cout << "cannot open reference the file";
		return -1;
	}

	char sentence[100];
	int line = 0;
	while (line != total_lines)
	{
		fin.getline(sentence, 100);
		sscanf_s(sentence, "%f %f %f", &reference_x->data.fl[line], &reference_y->data.fl[line], &reference_z->data.fl[line]);
		line++;
	}
	fin.close();
	return true;
}

bool read_deformed(CvMat* deformed_x, CvMat* deformed_y, CvMat* deformed_z, int total_lines)
{
	string path = "..\\deformed.txt";
	ifstream fin(path);
	if (!fin)
	{
		cout << "cannot open reference the file";
		return -1;
	}

	char sentence[100];
	int line = 0;
	while (line != total_lines)
	{
		fin.getline(sentence, 100);
		sscanf_s(sentence, "%f %f %f", &deformed_x->data.fl[line], &deformed_y->data.fl[line], &deformed_z->data.fl[line]);
		line++;
	}
	fin.close();
	return true;
}

bool GeometryCentroid(const CvMat* x, const CvMat* y, const CvMat* z, CvMat* centroid)
{
	float cx = cvAvg(x).val[0];
	float cy = cvAvg(y).val[0];
	float cz = cvAvg(z).val[0];

	CV_MAT_ELEM(*centroid, float, 0, 0) = cx;
	CV_MAT_ELEM(*centroid, float, 1, 0) = cy;
	CV_MAT_ELEM(*centroid, float, 2, 0) = cz;

	return true;
}

bool RelativeCoordinateToCentroid(const CvMat* px, const CvMat* py, const CvMat* pz, const CvMat* centroid, CvMat* relative)
{
	float cx = CV_MAT_ELEM(*centroid, float, 0, 0);
	float cy = CV_MAT_ELEM(*centroid, float, 1, 0);
	float cz = CV_MAT_ELEM(*centroid, float, 2, 0);

	int n = px->height;
	for (int i = 0; i < n; i++) {
		CV_MAT_ELEM(*relative, float, 0, i) = CV_MAT_ELEM(*px, float, i, 0) - cx;
		CV_MAT_ELEM(*relative, float, 1, i) = CV_MAT_ELEM(*py, float, i, 0) - cy;
		CV_MAT_ELEM(*relative, float, 2, i) = CV_MAT_ELEM(*pz, float, i, 0) - cz;
	}

	return true;
}
bool caculateRT(const CvMat* reference_x,
	const CvMat* reference_y,
	const CvMat* reference_z,
	const CvMat* deformed_x,
	const CvMat* deformed_y,
	const CvMat* deformed_z,
	CvMat* rotation,
	CvMat* translation)
{
	int point = reference_x->height;

	CvMat* reference_centroid = cvCreateMat(3, 1, CV_32F);
	CvMat* deformed_centroid = cvCreateMat(3, 1, CV_32F);

	GeometryCentroid(reference_x, reference_y, reference_z, reference_centroid);
	GeometryCentroid(deformed_x, deformed_y, deformed_z, deformed_centroid);

	CvMat* relative_reference = cvCreateMat(3, point, CV_32F);
	CvMat* relative_deformed = cvCreateMat(3, point, CV_32F);

	RelativeCoordinateToCentroid(reference_x, reference_y, reference_z, reference_centroid, relative_reference);
	RelativeCoordinateToCentroid(deformed_x, deformed_y, deformed_z, deformed_centroid, relative_deformed);

	CvMat* S = cvCreateMat(3, 3, CV_32F);
	cvGEMM(relative_reference, relative_deformed, 1, NULL, 0, S, CV_GEMM_B_T);

	/* S = U * E * VT */
	CvMat* UT = cvCreateMat(3, 3, CV_32F);
	CvMat* E = cvCreateMat(3, 3, CV_32F);
	CvMat* V = cvCreateMat(3, 3, CV_32F);

	cvSVD(S, E, UT, V, CV_SVD_U_T);

	CvMat* M = cvCreateMat(3, 3, CV_32F);
	cvGEMM(V, UT, 1, NULL, 0, M, CV_GEMM_B_T);

	CvMat* N = cvCreateMat(3, 3, CV_32F);
	cvSetIdentity(N);
	CV_MAT_ELEM(*N, float, 2, 2) = cvDet(M);

	cvMatMul(V, N, M);
	cvMatMul(M, UT, rotation);

	CvMat* L = cvCreateMat(3, 1, CV_32F);
	cvMatMul(rotation, reference_centroid, L);

	cvSub(deformed_centroid, L, translation);

	cvReleaseMat(&reference_centroid);
	cvReleaseMat(&deformed_centroid);
	cvReleaseMat(&relative_reference);
	cvReleaseMat(&relative_deformed);
	cvReleaseMat(&S);
	cvReleaseMat(&UT);
	cvReleaseMat(&E);
	cvReleaseMat(&V);
	cvReleaseMat(&M);
	cvReleaseMat(&N);
	cvReleaseMat(&L);

	return true;
}

bool write_RT(CvMat* R, CvMat* T)
{
	ofstream fout("..\\RT.txt");
	if (!fout)
	{
		cout << "cannot open reference the file";
		return -1;
	}

	fout << R->data.fl[0] << "\t" << R->data.fl[1] << "\t" << R->data.fl[2] << "\t" << T->data.fl[0] << "\n";
	fout << R->data.fl[3] << "\t" << R->data.fl[4] << "\t" << R->data.fl[5] << "\t" << T->data.fl[1] << "\n";
	fout << R->data.fl[6] << "\t" << R->data.fl[7] << "\t" << R->data.fl[8] << "\t" << T->data.fl[2] << "\n";
	return true;
}

int main()
{
	int reference_lines, deformed_lines;
	reference_lines = read_reference_lines() + 1;
	deformed_lines = read_deformed_lines() + 1;
	if (deformed_lines != reference_lines || reference_lines == -1)
		return -1;

	CvMat* reference_x = cvCreateMat(reference_lines, 1, CV_32F); 
	CvMat* reference_y = cvCreateMat(reference_lines, 1, CV_32F);
	CvMat* reference_z = cvCreateMat(reference_lines, 1, CV_32F);
	CvMat* deformed_x = cvCreateMat(deformed_lines, 1, CV_32F);
	CvMat* deformed_y = cvCreateMat(deformed_lines, 1, CV_32F);
	CvMat* deformed_z = cvCreateMat(deformed_lines, 1, CV_32F);

	read_reference(reference_x, reference_y, reference_z, reference_lines);
	read_deformed(deformed_x, deformed_y, deformed_z, deformed_lines);

	CvMat* R = cvCreateMat(3, 3, CV_32F);
	CvMat* T = cvCreateMat(3, 1, CV_32F);
	
	caculateRT(reference_x, reference_y, reference_z, deformed_x, deformed_y, deformed_z, R, T);
	write_RT(R,T);  // reference to deformed

	system("pause");
}
