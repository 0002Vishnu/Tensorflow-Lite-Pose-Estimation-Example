#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm> 
#include <cstdlib> // For std::atof

// Include TensorFlow Lite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

// Include OpenCV headers for image handling
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/trace.hpp>

using namespace tflite;
using namespace std;
using namespace cv;

// --- CONSTANTS ---
// **INPUT_SIZE for MoveNet Thunder (float32) is 256**
const int INPUT_SIZE = 256; 
const int CHANNELS = 3;
const int MAX_DISPLAY_DIM = 800; // Maximum dimension (width or height) for display
// -----------------

// Structure for a detected keypoint
struct Keypoint {
    float y;
    float x;
    float score;
    int id;
};

// Function definitions 
void draw_pose(Mat& image, const vector<Keypoint>& keypoints, float threshold, float scale_factor, float padding_x, float padding_y);

// Keypoint names 
const vector<string> KEYPOINT_NAMES = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

// Connect keypoint pairs (edges of the skeleton)
const vector<pair<int, int>> EDGES = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6},
    {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11},
    {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};


/**
 * @brief Runs TFLite inference using float32 input and output.
 * @param interpreter TFLite interpreter pointer.
 * @param image_input_float The 256x256 float32 RGB image (normalized 0.0-1.0) for input.
 * @return A vector of 17 Keypoint structs with float scores and coords.
 */
vector<Keypoint> run_movenet_inference(
    Interpreter* interpreter,
    const Mat& image_input_float) {
    
    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    
    // Check for correct input type (should be kTfLiteFloat32 for float model)
    if (input_tensor->type != kTfLiteFloat32) {
        cerr << "Error: Input tensor type is not Float32. Use the correct float model file." << endl;
        return {};
    }
    
    // Copy float data (in the range [0, 255])
    memcpy(input_tensor->data.f, image_input_float.data, INPUT_SIZE * INPUT_SIZE * CHANNELS * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        cerr << "Failed to invoke TFLite interpreter!" << endl;
        return {};
    }

    const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    // Read float output data directly (normalized [0.0, 1.0])
    const float* output_data = output_tensor->data.f;

    vector<Keypoint> keypoints;
    for (int i = 0; i < 17; ++i) {
        keypoints.push_back({
            output_data[i * 3 + 0], 
            output_data[i * 3 + 1], 
            output_data[i * 3 + 2], 
            i
        });
    }
    return keypoints;
}

void draw_pose(Mat& image, const vector<Keypoint>& keypoints, float threshold,
               float scale_factor, float padding_x, float padding_y) { 
    
    // The scale used to map from the INPUT_SIZE x INPUT_SIZE box back to original size
    const float drawing_scale = 1.0f / scale_factor;
    
    // Draw edges (Lines are Blue: Scalar(255, 0, 0) in BGR)
    for (const auto& edge : EDGES) {
        const Keypoint& kp1 = keypoints[edge.first];
        const Keypoint& kp2 = keypoints[edge.second];

        if (kp1.score > threshold && kp2.score > threshold) {
            
            // CRITICAL COORDINATE MAPPING (Aspect Ratio Correction)
            float x1_norm = kp1.x * (float)INPUT_SIZE;
            float y1_norm = kp1.y * (float)INPUT_SIZE;
            float x2_norm = kp2.x * (float)INPUT_SIZE;
            float y2_norm = kp2.y * (float)INPUT_SIZE;

            float x1_f = (x1_norm - padding_x) * drawing_scale;
            float y1_f = (y1_norm - padding_y) * drawing_scale;
            float x2_f = (x2_norm - padding_x) * drawing_scale;
            float y2_f = (y2_norm - padding_y) * drawing_scale;

            Point p1(static_cast<int>(x1_f), static_cast<int>(y1_f));
            Point p2(static_cast<int>(x2_f), static_cast<int>(y2_f));
            
            line(image, p1, p2, Scalar(255, 0, 0), 2);
        }
    }

    // Draw keypoints (Circles are Green: Scalar(0, 255, 0) in BGR)
    for (const auto& kp : keypoints) {
        if (kp.score > threshold) {
            
            // CRITICAL COORDINATE MAPPING
            float center_x_norm = kp.x * (float)INPUT_SIZE;
            float center_y_norm = kp.y * (float)INPUT_SIZE;

            float center_x_f = (center_x_norm - padding_x) * drawing_scale;
            float center_y_f = (center_y_norm - padding_y) * drawing_scale;

            Point center(static_cast<int>(center_x_f), static_cast<int>(center_y_f));
            
            circle(image, center, 5, Scalar(0, 255, 0), -1);
        }
    }
}


int main(int argc, char* argv[]) {
    cv::setNumThreads(1);
    
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <path_to_movenet_model.tflite> <path_to_input_image.jpg/png> <confidence_threshold_float>" << endl;
        cerr << "Example: " << argv[0] << " movenet_thunder_float.tflite image.jpg 0.5" << endl;
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];
    const float confidence_threshold = std::atof(argv[3]); 

    // 1. Load TFLite model
    unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_path);
    if (!model) { cerr << "Failed to load model: " << model_path << endl; return -1; }

    // 2. Build interpreter
    ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<Interpreter> interpreter;
    if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk ||
        interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "Failed to build or allocate TFLite interpreter." << endl; return -1;
    }

    // 3. Load the original image
    Mat image_original = imread(image_path);
    if (image_original.empty()) { cerr << "Failed to load image: " << image_path << endl; return -1; }

    // --- ASPECT RATIO CALCULATION & PREPROCESSING (Uses float32 logic) ---
    int orig_w = image_original.cols;
    int orig_h = image_original.rows;

    float scale_factor = std::min(static_cast<float>(INPUT_SIZE) / orig_w, 
                                  static_cast<float>(INPUT_SIZE) / orig_h);

    int scaled_w = static_cast<int>(orig_w * scale_factor);
    int scaled_h = static_cast<int>(orig_h * scale_factor);

    int padding_x = (INPUT_SIZE - scaled_w) / 2;
    int padding_y = (INPUT_SIZE - scaled_h) / 2;
    
    // 3a. Preprocessing: Resize/Pad the image for model input
    Mat image_resized_padded = Mat::zeros(INPUT_SIZE, INPUT_SIZE, CV_8UC3);
    Mat temp_resized;
    
    resize(image_original, temp_resized, Size(scaled_w, scaled_h));

    Rect roi(padding_x, padding_y, scaled_w, scaled_h);
    temp_resized.copyTo(image_resized_padded(roi));

    // Convert BGR to RGB
    Mat image_rgb;
    cvtColor(image_resized_padded, image_rgb, COLOR_BGR2RGB);

    // *******************************************************************
    // FIX: Convert to float32 and maintain the [0, 255] range as specified
    // *******************************************************************
    Mat image_input_float;
    image_rgb.convertTo(image_input_float, CV_32FC3, 1.0); // Scale factor is 1.0, not 1.0/255.0

    if (!image_input_float.isContinuous()) { image_input_float = image_input_float.clone(); }

    // 4. Run Inference and Get Keypoints (using float data)
    vector<Keypoint> keypoints = run_movenet_inference(interpreter.get(), image_input_float);

    if (keypoints.empty()) { cerr << "Pose detection failed." << endl; return -1; }

    // 5. Post-process and Visualize
    draw_pose(image_original, keypoints, confidence_threshold, scale_factor, (float)padding_x, (float)padding_y);

    // --- RESIZE FOR DISPLAY ---
    Mat image_display = image_original.clone();

    if (image_display.cols > MAX_DISPLAY_DIM || image_display.rows > MAX_DISPLAY_DIM) {
        float display_scale = std::min(static_cast<float>(MAX_DISPLAY_DIM) / image_display.cols,
                                       static_cast<float>(MAX_DISPLAY_DIM) / image_display.rows);
        
        Mat resized_for_display;
        resize(image_display, resized_for_display, Size(), display_scale, display_scale, INTER_AREA);
        image_display = resized_for_display;
    }

    // Display the resulting (potentially resized) image
    imshow("MoveNet Pose Detection (Float32 Thunder)", image_display);
    waitKey(0);

    return 0;
}