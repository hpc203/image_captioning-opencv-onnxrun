#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace Ort;

class Image_Caption
{
public:
	Image_Caption(string encoder_modelpath, string decoder_modelpath, string vocpath);
	string detect(Mat cv_image);

private:
	Net net;

	Mat normalize_(Mat img);
	const int inpWidth = 640;
	const int inpHeight = 640;
	int feat_len;
	int D;
	vector<float> input_tensor;
	float mean[3] = { 0.485, 0.456, 0.406 };
	float std[3] = { 0.229, 0.224, 0.225 };
	

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Image Caption");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs

	std::map<std::string, std::string> ix_to_word;
};

Image_Caption::Image_Caption(string encoder_modelpath, string decoder_modelpath, string vocpath)
{
	this->net = readNet(encoder_modelpath);

	std::wstring widestr = std::wstring(decoder_modelpath.begin(), decoder_modelpath.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->feat_len = input_node_dims[0][1];
	this->D = output_node_dims[0][1];

	ifstream infile_(vocpath);
	string line;
	while (getline(infile_, line))
	{
		size_t pos = line.find(":");
		string ix = line.substr(0, pos);
		const int len = line.length() - pos;
		string word = line.substr(pos + 1, len - 1); ///末尾的换行符,不要了
		ix_to_word[ix] = word;
	}
	infile_.close();
}

Mat Image_Caption::normalize_(Mat img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* std[c]), (0.0 - mean[c]) / std[c]);
	}
	Mat m_normalized_mat;
	merge(rgbChannels, m_normalized_mat);
	return m_normalized_mat;
}

string Image_Caption::detect(Mat srcimg)
{
	Mat temp_image;
	resize(srcimg, temp_image, cv::Size(this->inpWidth, this->inpHeight));
	Mat normalized_mat = this->normalize_(temp_image);
	Mat blob = blobFromImage(normalized_mat);
	this->net.setInput(blob);
	vector<Mat> outs;
	////net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	float* ptr_feat = (float*)outs[0].data;

	input_tensor.resize(this->feat_len);
	memcpy(&input_tensor[0], ptr_feat, this->feat_len * sizeof(float)); //内存拷贝
	array<int64_t, 2> input_shape_{ 1, this->feat_len };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_tensor.data(), input_tensor.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	const int64 *seq = ort_outputs[0].GetTensorMutableData<int64>();
	std::string words = "";
	for (int k = 0; k < this->D; k++)
	{
		if (seq[k] > 0)
		{
			std::string ix = std::to_string(seq[k]);
			if (words.length() > 0)
			{
				words += " ";
			}
			words += ix_to_word[ix];
		}
		else
		{
			break;
		}
	}
	return words;
}

int main()
{
	Image_Caption mynet("weights/encoder.onnx", "weights/decoder_fc_rl.onnx", "weights/vocab.txt");
	string imgpath = "testimgs/apple-490485_1920.jpg";
	Mat srcimg = imread(imgpath);
	string word = mynet.detect(srcimg);

	cout << word << endl;
	putText(srcimg, word, Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	static const string kWinName = "Deep learning Image Caption in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}