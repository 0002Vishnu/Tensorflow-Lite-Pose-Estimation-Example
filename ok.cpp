#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <cmath>

// --- OpenCV Headers ---
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

// --- TensorFlow Lite Headers ---
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h" 
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/mutable_op_resolver.h"
// CRITICAL FIX: Include the specific kernel header to guarantee op function definition
#include "tensorflow/lite/kernels/builtin_op_kernels.h" 

using namespace tflite;
using namespace cv;

// --- Constants ---
const int INPUT_H = 256;
const int INPUT_W = 256;
const float CONFIDENCE_THRESHOLD = 0.3f;
const int NUM_KEYPOINTS = 17;
const int INPUT_TENSOR_INDEX = 0;

// Movenet Keypoint structure (Y, X, Confidence)
struct Keypoint {
    float y;
    float x;
    float confidence;
    Point point; // OpenCV point in original image coordinates
};

// --- TFLite Setup and Inference ---

/**
 * Loads the TFLite model and sets up the interpreter.
 * Uses MutableOpResolver with a comprehensive list of manually registered built-in operators.
 */
std::unique_ptr<Interpreter> LoadModelAndSetup(const char* model_path) {
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "ERROR: Failed to load model: " << model_path << std::endl;
        return nullptr;
    }

    // Use MutableOpResolver and manually register ALL built-in ops to fix dependency issues.
    







    


    // *** CRITICAL FIX: Use MutableOpResolver and manually register ALL ops ***
    tflite::MutableOpResolver resolver;
    
    // FINAL, EXHAUSTIVE LIST covering all built-in operators found in MoveNet (including quantization)
    resolver.AddBuiltin(BuiltinOperator_ABS, tflite::ops::builtin::Register_ABS(), 1);
    resolver.AddBuiltin(BuiltinOperator_ADD, tflite::ops::builtin::Register_ADD(), 1);
    resolver.AddBuiltin(BuiltinOperator_ARG_MAX, tflite::ops::builtin::Register_ARG_MAX(), 1);
    resolver.AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, tflite::ops::builtin::Register_AVERAGE_POOL_2D(), 1);
    resolver.AddBuiltin(BuiltinOperator_BROADCAST_TO, tflite::ops::builtin::Register_BROADCAST_TO(), 1);
    resolver.AddBuiltin(BuiltinOperator_CAST, tflite::ops::builtin::Register_CAST(), 1);
    resolver.AddBuiltin(BuiltinOperator_CONCATENATION, tflite::ops::builtin::Register_CONCATENATION(), 1);
    resolver.AddBuiltin(BuiltinOperator_CONV_2D, tflite::ops::builtin::Register_CONV_2D(), 1);
    
    // *** FIX FOR DEQUANTIZE ***
    // Note: We force version 2 if the model specifically requested it, but for built-in ops, version 1 often works.
    resolver.AddBuiltin(BuiltinOperator_DEQUANTIZE, tflite::ops::builtin::Register_DEQUANTIZE(), 2); 
    
    resolver.AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(), 1);
    resolver.AddBuiltin(BuiltinOperator_DIV, tflite::ops::builtin::Register_DIV(), 1);
    resolver.AddBuiltin(BuiltinOperator_EQUAL, tflite::ops::builtin::Register_EQUAL(), 1);
    resolver.AddBuiltin(BuiltinOperator_EXP, tflite::ops::builtin::Register_EXP(), 1); 
    resolver.AddBuiltin(BuiltinOperator_EXPAND_DIMS, tflite::ops::builtin::Register_EXPAND_DIMS(), 1);
    resolver.AddBuiltin(BuiltinOperator_FILL, tflite::ops::builtin::Register_FILL(), 1);
    resolver.AddBuiltin(BuiltinOperator_FLOOR, tflite::ops::builtin::Register_FLOOR(), 1);
    resolver.AddBuiltin(BuiltinOperator_FLOOR_DIV, tflite::ops::builtin::Register_FLOOR_DIV(), 1);
    resolver.AddBuiltin(BuiltinOperator_GATHER, tflite::ops::builtin::Register_GATHER(), 1);
    resolver.AddBuiltin(BuiltinOperator_GATHER_ND, tflite::ops::builtin::Register_GATHER_ND(), 1); 
    resolver.AddBuiltin(BuiltinOperator_GREATER, tflite::ops::builtin::Register_GREATER(), 1);
    resolver.AddBuiltin(BuiltinOperator_GREATER_EQUAL, tflite::ops::builtin::Register_GREATER_EQUAL(), 1);
    resolver.AddBuiltin(BuiltinOperator_HARD_SWISH, tflite::ops::builtin::Register_HARD_SWISH(), 1);
    resolver.AddBuiltin(BuiltinOperator_LEAKY_RELU, tflite::ops::builtin::Register_LEAKY_RELU(), 1);
    resolver.AddBuiltin(BuiltinOperator_LESS, tflite::ops::builtin::Register_LESS(), 1);
    resolver.AddBuiltin(BuiltinOperator_LESS_EQUAL, tflite::ops::builtin::Register_LESS_EQUAL(), 1);
    resolver.AddBuiltin(BuiltinOperator_LOGICAL_AND, tflite::ops::builtin::Register_LOGICAL_AND(), 1);
    resolver.AddBuiltin(BuiltinOperator_LOGISTIC, tflite::ops::builtin::Register_LOGISTIC(), 1);
    resolver.AddBuiltin(BuiltinOperator_MAXIMUM, tflite::ops::builtin::Register_MAXIMUM(), 1);
    resolver.AddBuiltin(BuiltinOperator_MAX_POOL_2D, tflite::ops::builtin::Register_MAX_POOL_2D(), 1);
    resolver.AddBuiltin(BuiltinOperator_MEAN, tflite::ops::builtin::Register_MEAN(), 1); 
    resolver.AddBuiltin(BuiltinOperator_MINIMUM, tflite::ops::builtin::Register_MINIMUM(), 1);
    resolver.AddBuiltin(BuiltinOperator_MUL, tflite::ops::builtin::Register_MUL(), 1);
    resolver.AddBuiltin(BuiltinOperator_PACK, tflite::ops::builtin::Register_PACK(), 1);
    resolver.AddBuiltin(BuiltinOperator_PAD, tflite::ops::builtin::Register_PAD(), 1);
    resolver.AddBuiltin(BuiltinOperator_POW, tflite::ops::builtin::Register_POW(), 1); 
    resolver.AddBuiltin(BuiltinOperator_QUANTIZE, tflite::ops::builtin::Register_QUANTIZE(), 1); // Safety for input/output
    resolver.AddBuiltin(BuiltinOperator_RANGE, tflite::ops::builtin::Register_RANGE(), 1);
    resolver.AddBuiltin(BuiltinOperator_REDUCE_MAX, tflite::ops::builtin::Register_REDUCE_MAX(), 1);
    resolver.AddBuiltin(BuiltinOperator_REDUCE_PROD, tflite::ops::builtin::Register_REDUCE_PROD(), 1); 
    resolver.AddBuiltin(BuiltinOperator_RELU, tflite::ops::builtin::Register_RELU(), 1);
    
    // RESIZE_BILINEAR fix (forcing version 3)
    resolver.AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, tflite::ops::builtin::Register_RESIZE_BILINEAR(), 3);
    
    resolver.AddBuiltin(BuiltinOperator_RESHAPE, tflite::ops::builtin::Register_RESHAPE(), 1);
    resolver.AddBuiltin(BuiltinOperator_RSQRT, tflite::ops::builtin::Register_RSQRT(), 1);
    resolver.AddBuiltin(BuiltinOperator_SELECT, tflite::ops::builtin::Register_SELECT(), 1);
    resolver.AddBuiltin(BuiltinOperator_SHAPE, tflite::ops::builtin::Register_SHAPE(), 1);
    resolver.AddBuiltin(BuiltinOperator_SQUARE, tflite::ops::builtin::Register_SQUARE(), 1);
    resolver.AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, tflite::ops::builtin::Register_SQUARED_DIFFERENCE(), 1);
    resolver.AddBuiltin(BuiltinOperator_SQUEEZE, tflite::ops::builtin::Register_SQUEEZE(), 1);
    resolver.AddBuiltin(BuiltinOperator_SPLIT, tflite::ops::builtin::Register_SPLIT(), 1);
    resolver.AddBuiltin(BuiltinOperator_STRIDED_SLICE, tflite::ops::builtin::Register_STRIDED_SLICE(), 1);
    resolver.AddBuiltin(BuiltinOperator_SUB, tflite::ops::builtin::Register_SUB(), 1);
    resolver.AddBuiltin(BuiltinOperator_SUM, tflite::ops::builtin::Register_SUM(), 1);
    resolver.AddBuiltin(BuiltinOperator_TANH, tflite::ops::builtin::Register_TANH(), 1);
    resolver.AddBuiltin(BuiltinOperator_TILE, tflite::ops::builtin::Register_TILE(), 1);
    resolver.AddBuiltin(BuiltinOperator_TOPK_V2, tflite::ops::builtin::Register_TOPK_V2(), 1);
    resolver.AddBuiltin(BuiltinOperator_UNPACK, tflite::ops::builtin::Register_UNPACK(), 1);
    resolver.AddBuiltin(BuiltinOperator_ZEROS_LIKE, tflite::ops::builtin::Register_ZEROS_LIKE(), 1);

    std::unique_ptr<Interpreter> interpreter;
// ... (rest of the LoadModelAndSetup function)







// ... (rest of the LoadModelAndSetup function)
    InterpreterBuilder(*model, resolver)(&interpreter); 
    if (!interpreter) {
        std::cerr << "ERROR: Failed to construct interpreter. (InterpreterBuilder failed)" << std::endl;
        return nullptr;
    }

    // Crucial: NO DELEGATE CODE (Ensures fallback to generic CPU kernels and avoids dynamic tensor crash)

    if (interpreter->inputs().empty() || interpreter->inputs()[0] != INPUT_TENSOR_INDEX) {
        std::cerr << "ERROR: Model structure unexpected. Input not at index 0." << std::endl;
        return nullptr;
    }
    
    std::cout << "INFO: Using input tensor index: " << INPUT_TENSOR_INDEX << std::endl;

    interpreter->SetNumThreads(4);
    
    if (interpreter->ResizeInputTensor(INPUT_TENSOR_INDEX, {1, INPUT_H, INPUT_W, 3}) != kTfLiteOk) {
        std::cerr << "ERROR: Failed to resize input tensor." << std::endl;
        return nullptr;
    }









    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Failed to allocate tensors." << std::endl;
        return nullptr;
    }
    
    if (interpreter->tensor(INPUT_TENSOR_INDEX)->data.f == nullptr) {
         std::cerr << "ERROR: Input tensor 0 data is NULL even after allocation." << std::endl;
         return nullptr;
    }
    
    return interpreter;
}

/**
 * Preprocesses the image for the Movenet model.
 * Loads the image, resizes it, and converts it to a normalized float tensor.
 */






// In src/main.cpp:
// In src/main.cpp:
bool PreprocessImage(const std::string& image_path, const Mat& original_image, Interpreter* interpreter) {
    Mat resized_image;
    resize(original_image, resized_image, Size(INPUT_W, INPUT_H), 0, 0, INTER_LINEAR);

    // --- FIX 1: Convert BGR to RGB and ensure 8-bit unsigned char (CV_8UC3) ---
    Mat rgb_uint8_image;
    // TFLite models often expect RGB, but the data needs to be 8-bit integers (0-255)
    cvtColor(resized_image, rgb_uint8_image, COLOR_BGR2RGB); 
    
    // Ensure the matrix is of type CV_8UC3, which corresponds to uint8_t
    if (rgb_uint8_image.type() != CV_8UC3) {
        rgb_uint8_image.convertTo(rgb_uint8_image, CV_8UC3);
    }
    // --- END FIX 1 ---

    TfLiteTensor* input_tensor = interpreter->tensor(INPUT_TENSOR_INDEX);
    if (!input_tensor || input_tensor->data.data == nullptr) {
        std::cerr << "ERROR: Failed to access input tensor data structure." << std::endl;
        return false;
    }
    
    // Check tensor type and copy data accordingly
    if (input_tensor->type == kTfLiteUInt8) {
        // --- FIX 2: Handle 8-bit unsigned integer input type ---
        uint8_t* input_tensor_data = input_tensor->data.uint8;
        size_t data_size = INPUT_W * INPUT_H * 3 * sizeof(uint8_t);
        memcpy(input_tensor_data, rgb_uint8_image.data, data_size);
        std::cout << "INFO: Successfully loaded image as 8-bit integer tensor." << std::endl;
        // --- END FIX 2 ---
        
    } else if (input_tensor->type == kTfLiteFloat32) {
        // Fallback for float32 (if you ever switch models)
        Mat rgb_float_image;
        resized_image.convertTo(rgb_float_image, CV_32FC3, 1.0 / 255.0); // Normalize to 0-1
        cvtColor(rgb_float_image, rgb_float_image, COLOR_BGR2RGB);

        float* input_tensor_data = input_tensor->data.f;
        size_t data_size = INPUT_W * INPUT_H * 3 * sizeof(float);
        memcpy(input_tensor_data, rgb_float_image.data, data_size);
        std::cout << "INFO: Successfully loaded image as float32 tensor." << std::endl;
        
    } else {
        std::cerr << "ERROR: Input tensor has unexpected data type: " << input_tensor->type << std::endl;
        return false;
    }
    
    return true;
}







/**
 * Post-processes the model output to extract keypoints in original image coordinates.
 */
std::vector<Keypoint> PostProcessOutput(Interpreter* interpreter, int original_w, int original_h) {
    const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    if (!output_tensor) {
        std::cerr << "ERROR: Failed to get output tensor." << std::endl;
        return {};
    }

    // Movenet output shape is typically [1, 1, 17, 3] (Y, X, Confidence) for single pose,
    // or [1, N, 56] (N poses * 17 keypoints * 3 values + NMS scores) for multipose.
    // For simplicity, we are still parsing the first pose as per the original setup.
    const float* output_data = interpreter->typed_output_tensor<float>(0);
    
    // Check if output is the multi-pose tensor shape [1, N, 56]
    int N = output_tensor->dims->data[1]; // Number of detections


    if (N <= 0) {
    // CRITICAL: Handle the case where no poses were detected gracefully
    std::cout << "WARNING: No poses detected (N=0)." << std::endl;
    return {}; // Return empty vector instead of crashing
    }


    if (N > 0 && output_tensor->dims->data[2] == (NUM_KEYPOINTS * 3 + 1)) {
        std::cout << "INFO: Multi-pose output detected (" << N << " poses). Only processing the first pose." << std::endl;
        // Output for the first pose starts at index 0 and contains 56 elements.
        
        std::vector<Keypoint> keypoints(NUM_KEYPOINTS);
        // The keypoints are elements 1 to 52 (51: 17 * 3). We ignore the first element (box score) and last three (box coordinates).
        for (int i = 0; i < NUM_KEYPOINTS; ++i) {
            // Keypoint data starts after the box score (index 0).
            // Keypoint i data starts at index 1 + i*3
            
            // NOTE: Multi-pose output structure is complex and varies. For the common format (normalized keypoints):
            // The 56 elements are [score, ymin, xmin, ymax, xmax, keypoint_0_y, keypoint_0_x, keypoint_0_score, ..., keypoint_16_score]
            
            // Assuming simplified output for the first pose:
            int base_idx = 5; // Start index of the keypoint data after box coords
            int kp_idx = base_idx + i * 3;

            keypoints[i].y = output_data[kp_idx];
            keypoints[i].x = output_data[kp_idx + 1];
            keypoints[i].confidence = output_data[kp_idx + 2];
            
            // Denormalize the keypoints back to the original image size
            keypoints[i].point.x = static_cast<int>(keypoints[i].x * original_w);
            keypoints[i].point.y = static_cast<int>(keypoints[i].y * original_h);
        }
        return keypoints;

    } else if (output_tensor->dims->data[2] == 3) {
        std::cout << "INFO: Single-pose output detected (17 keypoints)." << std::endl;
        // Single pose output [1, 1, 17, 3] (Y, X, Confidence)
        std::vector<Keypoint> keypoints(NUM_KEYPOINTS);

        for (int i = 0; i < NUM_KEYPOINTS; ++i) {
            keypoints[i].y = output_data[i * 3 + 0];
            keypoints[i].x = output_data[i * 3 + 1];
            keypoints[i].confidence = output_data[i * 3 + 2];

            // Denormalize the keypoints back to the original image size
            keypoints[i].point.x = static_cast<int>(keypoints[i].x * original_w);
            keypoints[i].point.y = static_cast<int>(keypoints[i].y * original_h);
        }
        return keypoints;
    }
    
    std::cerr << "WARNING: Unknown output tensor shape. Cannot process pose results." << std::endl;
    return {};
}

/**
 * Draws the detected pose skeleton onto the original image.
 */
void DrawPoses(Mat& image, const std::vector<Keypoint>& keypoints) {
    if (keypoints.empty()) return;

    // Movenet COCO Keypoint IDs and Connections:
    // 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear, 5:left_shoulder, 
    // 6:right_shoulder, 7:left_elbow, 8:right_elbow, 9:left_wrist, 10:right_wrist, 
    // 11:left_hip, 12:right_hip, 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
    
    // Define connections (pairs of keypoint indices)
    std::vector<std::pair<int, int>> connections = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {11, 12}, {5, 11}, {6, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    // Draw connections
    for (const auto& conn : connections) {
        const Keypoint& kp1 = keypoints[conn.first];
        const Keypoint& kp2 = keypoints[conn.second];

        if (kp1.confidence > CONFIDENCE_THRESHOLD && kp2.confidence > CONFIDENCE_THRESHOLD) {
            // Use blue for the connection line
            line(image, kp1.point, kp2.point, Scalar(255, 0, 0), 2);
        }
    }

    // Draw keypoints
    for (const auto& kp : keypoints) {
        if (kp.confidence > CONFIDENCE_THRESHOLD) {
            // Use green for keypoints
            circle(image, kp.point, 5, Scalar(0, 255, 0), -1);
        }
    }
}


// --- Main Function ---

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <tflite_model_path> <image_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const std::string image_path = argv[2];

    // 1. Load Model and Setup Interpreter
    std::unique_ptr<Interpreter> interpreter = LoadModelAndSetup(model_path);
    if (!interpreter) {
        std::cerr << "Exiting due to setup failure." << std::endl;
        return 1;
    }

    // 2. Load and Preprocess Image
    Mat original_image = imread(image_path);
    if (original_image.empty()) {
        std::cerr << "ERROR: Failed to load image: " << image_path << std::endl;
        return 1;
    }

    int original_w = original_image.cols;
    int original_h = original_image.rows;

    if (!PreprocessImage(image_path, original_image, interpreter.get())) {
        std::cerr << "Exiting due to preprocessing failure." << std::endl;
        return 1;
    }

    // 3. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "ERROR: Failed to invoke interpreter." << std::endl;
        return 1;
    }
    
    std::cout << "INFO: Inference successful." << std::endl;

    // 4. Post-process and Draw Results
    std::vector<Keypoint> keypoints = PostProcessOutput(interpreter.get(), original_w, original_h);
    
    DrawPoses(original_image, keypoints);

    // 5. Display Result
    const std::string output_window_name = "Pose Estimation Result";
    namedWindow(output_window_name, WINDOW_AUTOSIZE);
    imshow(output_window_name, original_image);
    std::cout << "INFO: Press any key in the display window to exit." << std::endl;
    waitKey(0);

    return 0;
}
