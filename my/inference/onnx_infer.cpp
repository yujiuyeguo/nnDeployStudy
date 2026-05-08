#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDefaultObsDim = 36;

std::vector<int64_t> NormalizeShape(const std::vector<int64_t>& raw_shape) {
    std::vector<int64_t> shape = raw_shape;
    if (shape.empty()) {
        return {1, kDefaultObsDim};
    }
    for (auto& dim : shape) {
        if (dim <= 0) {
            dim = 1;
        }
    }
    return shape;
}

size_t ElementCount(const std::vector<int64_t>& shape) {
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

void PrintShape(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 != shape.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string model_path =
        argc > 1 ? argv[1] : "/home/cy/onnx_study/my/model/policy.onnx";

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "isaaclab_infer");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        const size_t input_count = session.GetInputCount();
        const size_t output_count = session.GetOutputCount();

        if (input_count == 0 || output_count == 0) {
            std::cerr << "Model has no inputs or outputs." << std::endl;
            return 1;
        }

        std::vector<std::string> input_name_store;
        std::vector<const char*> input_names;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<float>> input_buffers;
        std::vector<Ort::Value> input_tensors;

        std::cout << "Loaded model: " << model_path << std::endl;
        std::cout << "Inputs:" << std::endl;
        for (size_t i = 0; i < input_count; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto raw_shape = type_info.GetShape();
            auto shape = NormalizeShape(raw_shape);
            const size_t element_count = ElementCount(shape);

            input_name_store.emplace_back(name.get());
            input_names.push_back(input_name_store.back().c_str());
            input_shapes.push_back(shape);
            input_buffers.emplace_back(element_count, 1.0f);

            std::cout << "  " << input_name_store.back() << " shape=";
            PrintShape(raw_shape);
            std::cout << " normalized=";
            PrintShape(shape);
            std::cout << " elements=" << element_count << std::endl;
        }

        std::vector<std::string> output_name_store;
        std::vector<const char*> output_names;

        std::cout << "Outputs:" << std::endl;
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto type_info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();

            output_name_store.emplace_back(name.get());
            output_names.push_back(output_name_store.back().c_str());

            std::cout << "  " << output_name_store.back() << " shape=";
            PrintShape(shape);
            std::cout << std::endl;
        }

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (size_t i = 0; i < input_count; ++i) {
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                input_buffers[i].data(),
                input_buffers[i].size(),
                input_shapes[i].data(),
                input_shapes[i].size()));
        }

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());

        std::cout << "Inference finished." << std::endl;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            const float* data = output_tensors[i].GetTensorData<float>();
            const size_t element_count = ElementCount(NormalizeShape(shape));
            const size_t preview_count = std::min<size_t>(element_count, 8);

            std::cout << "Output " << output_name_store[i] << " shape=";
            PrintShape(shape);
            std::cout << " preview=[";
            for (size_t j = 0; j < preview_count; ++j) {
                std::cout << data[j];
                if (j + 1 != preview_count) {
                    std::cout << ", ";
                }
            }
            if (element_count > preview_count) {
                std::cout << ", ...";
            }
            std::cout << "]" << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
