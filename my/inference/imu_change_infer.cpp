#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// 一条 IMU 样本，对应 csv 文件中的一行。
struct ImuSample {
    int64_t timestamp_ms = 0;
    std::array<float, 6> features{};
    int label = -1;
};

// 把模型输出的 logit 转成 0~1 概率。
float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 打印命令行用法说明。
void PrintUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [model.onnx] [imu.csv]\n\n"
        << "Default model:\n"
        << "  /home/cy/onnx_study/my/imu_change_detector.onnx\n\n"
        << "Default csv:\n"
        << "  /home/cy/onnx_study/my/imu_change_test.csv\n";
}

// 按逗号切分一行 csv。
std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

// 把一行 csv 文本解析成 ImuSample 结构体。
// 列顺序要求为：
// timestamp_ms, roll_deg, pitch_deg, yaw_deg, gyro_x_dps, gyro_y_dps, gyro_z_dps, label
bool ParseSample(const std::string& line, ImuSample& sample) {
    const auto fields = SplitCsvLine(line);
    if (fields.size() < 8) {
        return false;
    }

    sample.timestamp_ms = std::stoll(fields[0]);
    sample.features[0] = std::stof(fields[1]);
    sample.features[1] = std::stof(fields[2]);
    sample.features[2] = std::stof(fields[3]);
    sample.features[3] = std::stof(fields[4]);
    sample.features[4] = std::stof(fields[5]);
    sample.features[5] = std::stof(fields[6]);
    sample.label = std::stoi(fields[7]);
    return true;
}

// 单次推理核心函数：
// 1. 把 6 轴 IMU 数据组织成 [1, 6] 的输入张量
// 2. 调用 ONNX Runtime 执行一次前向推理
// 3. 返回模型原始输出 logit
float RunSingleInference(


    Ort::Session& session,

    /*
    因为一个 ONNX 模型内部的输入输出都是有名字的，
    ONNX模型导出的每一个输入和输出节点都有明确的名字。这就像这个节点的“身份证”，让后续的推理引擎能够准确、高效地读写数据，也是对模型进行优化和修改的基础。


    在run 的时候要导入 ， 告诉 ONNX Runtime，这个输入张量要喂给模型里的哪个输入节点，以及要取哪个输出节点”。
    */
    const char* input_name,
    const char* output_name,

    const std::array<float, 6>& features) {
    // 这个模型的输入是二维张量 [batch, feature_dim]。
    // 当前一次只推理 1 条数据，所以 batch=1；每条样本有 6 个特征，所以是 [1, 6]。
    std::vector<int64_t> input_shape = {1, 6};

    // 把 6 轴 IMU 数据整理成一段连续的 float 内存。
    // ONNX Runtime 创建 Tensor 时，需要能拿到这段连续内存的首地址。
    std::vector<float> input_buffer(features.begin(), features.end());
    /*
        基本步骤
        1：create cpu 
        
        2：create input tensor 

        3：create output tensor
    */


    // 必须的 ， 没有 MemoryInfo	❌ 编译失败	CreateTensor 强制要求该参数
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    /*ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));


    1:OrtDeviceAllocator：每次分配都走系统调用，频繁分配时效率较低，但内存释放更及时 , OrtArenaAllocator：适合频繁分配/释放小内存的场景（如循环推理），但可能占用更多内存

    2:OrtMemTypeCPU 常与 OrtMemTypeCPUOutput 配对使用（后者用于输出张量的优化），但单个使用时没区别

    3: std::make_unique<Ort::MemoryInfo>(...)	直接栈对象 Ort::MemoryInfo memory_info = ... , 作为函数返回

    但如果你追求性能（尤其是高频推理场景），建议把 OrtDeviceAllocator 改为 OrtArenaAllocator。至于 unique_ptr 包装，除非确实需要动态生命周期，否则直接作为成员变量更简洁。

    */

    

    // 输入 tensor 的创建
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_buffer.data(),
        input_buffer.size(),
        input_shape.data(),
        input_shape.size());

    /*
        CreateTensor(...) 这种“创建输入 Tensor”的动作是必须的 ， 把 buffer 的数据“映射”或“包装”成指定 shape 的张量 ，告诉 ONNX Runtime "这段内存请按照 shape 的形状来解读"。

        ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, 
         ctx->input_buffer.data(),
        ctx->input_buffer.size(), 
        ctx->input_shape.data(), 
        ctx->input_shape.size()));

        关键是最后要给 ONNX Runtime：
        数据指针
        元素个数
        shape 指针
        shape 维度数

        参数解释：
        memory_info: 数据所在的设备和内存类型，这里是 CPU
        input_buffer.data(): 输入数据首地址
        input_buffer.size(): 输入元素个数，这里是 6
        input_shape.data(): 形状数组首地址，这里是 [1, 6]
        input_shape.size(): 形状维度数，这里是 2
    */

    // 应该还有输出 tensor 的创建

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};



    /*

    // 场景1：在线服务（多Session并发）
    session_options.DisablePerSessionThreads();  // ✅ 推荐
    默认行为：每个Session会创建自己的线程池
    设置后：使用全局线程池，所有Session共享 
    优点：减少线程创建开销，节省内存
    适用：多个Session并发运行（如同时加载多个模型）


    session_options.EnableCpuMemArena();         // ✅ 推荐
    作用：预分配一大块内存池，重复使用而不是频繁malloc/free
    优点：减少内存分配开销，提升推理速度（尤其小模型）
    缺点：内存占用略高（预分配策略）
    建议：生产环境通常开启


    session_options.EnableMemPattern();          // ✅ 推荐

    作用：首次推理时记录内存分配模式，后续复用相同模式
    优点：减少内存碎片，加速后续推理
    原理：类似"内存使用模板"（memory pattern）
    注意：需配合EnableCpuMemArena()使用效果更好

    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL); // 便于调试
    
    ORT_ENABLE_ALL：启用所有优化（常数折叠、算子融合等）

    其他选项：

        ORT_DISABLE_ALL：无优化（调试用）

        ORT_ENABLE_BASIC：基础优化

        ORT_ENABLE_EXTENDED：扩展优化

    效果：最快推理速度，但首次加载稍慢


    对于内存受限的场景
    session_options.EnableCpuMemArena();         // ⚠️ 可能内存占用高
    // 考虑关闭CpuMemArena，根据实际测试决定



    
    推理框架的搭建
    
    1：配置ONNX Runtime会话（Session）的运行选项，最大化CPU推理性能

    2：动态的获取 输入的 name 和 shape

    3: 将 cpu的memory的地址传入，用于创建输入和输出的 tensor
    
    */



    // 调用 Session::Run 执行一次真正的 ONNX 推理。
    //
    // 可以把 session 理解成“已经加载好的模型对象”，
    // 把 input_tensor 理解成“这次送给模型的一条样本”，
    // Run(...) 的结果就是“模型输出的 Tensor 列表”。
    //
    // 参数解释：
    // - Ort::RunOptions{nullptr}: 本次推理的运行配置，这里使用默认值
    // - input_names: 输入节点名数组，必须和模型图中的输入名一致
    // - &input_tensor: 输入 Tensor 数组首地址
    // - 1: 输入 Tensor 的个数，这里只有 1 个输入
    // - output_names: 希望取回的输出节点名数组
    // - 1: 输出节点个数，这里只有 1 个输出
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    // 当前模型输出 shape 是 [1, 1]，含义是：
    // - batch=1，对应 1 条样本
    // - 每条样本输出 1 个数值，也就是 change_logit
    //
    // 所以这里直接取第 0 个输出 Tensor 中的第 0 个 float 即可。
    return output_tensors[0].GetTensorData<float>()[0];
}



}  // namespace



int main(int argc, char* argv[]) {
    const std::string default_model_path = "/home/cy/onnx_study/my/imu_change_detector.onnx";
    const std::string default_csv_path = "/home/cy/onnx_study/my/imu_change_test.csv";

    if (argc > 3) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string model_path = argc >= 2 ? argv[1] : default_model_path;
    const std::string csv_path = argc >= 3 ? argv[2] : default_csv_path;

    try {
        // 主框架 1：
        // 打开 IMU 数据文件。这里的数据源目前是 csv，
        // 后面如果你要改成串口、socket、共享内存，主要改这里和下面的读取循环。
        std::ifstream input_file(csv_path);
        if (!input_file.is_open()) {
            std::cerr << "Failed to open IMU csv file: " << csv_path << std::endl;
            return 1;
        }

        /*


        主框架 2：
        初始化 ONNX Runtime，并加载导出的 imu_change_detector.onnx 模型。
        整个 ONNX 推理部分的主流程基本就是：
        1. 创建 Env
        2. 创建并配置 SessionOptions ， 这是配置选项
        3. 通过 onnx 文件创建 Session
        4. 查询模型输入输出名
        5. 构造输入 输出 Tensor
        6. 调用 session.Run()
        7. 读取输出 Tensor 内容

        */

        /*
       Ort::Env是ONNX Runtime的"总引擎"，必须在所有Session之前创建且只创建一次。它管理全局资源（线程池、日志、内存分配器），配置合理与否直接影响整个推理系统的性能。

        Ort::ThreadingOptions thread_opts;

        条件判断作用， 用户指定了线程数才设置如果：

        intra_threads_ <= 0，不调用设置函数
        ONNX Runtime会使用默认值（通常是CPU核心数）
        让用户可以选择"让ORT自动决定"

        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }

        这段代码通过条件判断，实现了灵活的线程控制，既支持用户手动调优，又保留ORT自动优化的能力，是生产环境常用的设计模式。




            env_ = std::make_unique<Ort::Env>(
            thread_opts,                      // 线程选项配置
            ORT_LOGGING_LEVEL_WARNING,        // 日志级别
            "ONNXRuntimeInference"           // 环境名称
        

        thread_opts.SetGlobalIntraOpNumThreads(4);  // 算子内 使用4个线程 算的更快 ， 就使用这个吧
        thread_opts.SetGlobalInterOpNumThreads(2);  // 算子间 并行

        intra_op vs inter_op
        先理解两个概念：
        术语	含义	作用
        Intra-op	算子内并行	单个算子（如矩阵乘法）内部使用多线程
        Inter-op	算子间并行	多个不同算子同时并行执行

        1：thread_opts 作用：配置全局线程池
        设置并行线程数量
        设置线程亲和性（绑定CPU核心）
        设置线程优先级


        线程设置策略：
        // 场景1：用户指定
        intra_threads_ = 4;  // 使用4线程
        → SetGlobalIntraOpNumThreads(4)

        // 场景2：用户未指定/自动
        intra_threads_ = 0;  // 跳过设置
        → ORT自动选择（通常 = CPU核心数）

        // 场景3：单线程模式
        intra_threads_ = 1;  // 强制单线程
        → SetGlobalIntraOpNumThreads(1）



        2: ORT_LOGGING_LEVEL_WARNING - 日志级别 
        级别	用途	输出内容
        ORT_LOGGING_LEVEL_VERBOSE	调试	所有细节（大量输出）
        ORT_LOGGING_LEVEL_INFO	开发	一般信息
        ORT_LOGGING_LEVEL_WARNING	生产	仅警告和错误 ✅
        ORT_LOGGING_LEVEL_ERROR	生产	仅错误
        ORT_LOGGING_LEVEL_FATAL	生产	仅致命错误

        3. "ONNXRuntimeInference" - 环境名称

        作用：标识日志来源（多环境时区分）
        示例输出：
        [ONNXRuntimeInference:WARNING] ... 警告信息
        [ONNXRuntimeInference:ERROR] ... 错误信息

        特点:
        一个进程只能有一个Ort::Env实例
        多个Session共享同一个Env
        生命周期贯穿整个程序




    线程数设置建议：

    // 强化学习通常batch_size=1，单线程足够
    thread_opts.SetGlobalIntraOpNumThreads(1);
    // 如果需要并行推理多个环境（如A2C多worker）
    thread_opts.SetGlobalIntraOpNumThreads(4);

    日志级别建议：

        开发调试：ORT_LOGGING_LEVEL_INFO（看模型加载信息）

        生产环境：ORT_LOGGING_LEVEL_WARNING（避免刷日志）  一般来说是这个

        性能测试：ORT_LOGGING_LEVEL_ERROR（最小开销）

    常见错误：

        忘记创建Env直接创建Session → 崩溃

        多次创建Env → 运行时异常

        Env生命周期短于Session → 未定义行为

        */

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "imu_change_infer");



        /*
            SessionOptions 用来控制推理时的一些行为，比如线程数、图优化等级等。 如果以后你要接 CPU/GPU、多线程优化，通常也是从这里继续扩展。
        */
        Ort::SessionOptions session_options;

        /*
            session option 配置


            // 全局配置（Env级别）
            thread_opts.SetGlobalIntraOpNumThreads(4);  // 影响所有Session

            // 会话配置（Session级别）- 会覆盖全局设置
            session_options.SetIntraOpNumThreads(2);    // 该Session用2线程
            设置算子内部使用的线程数。 这里设为 1，便于观察和调试，逻辑最简单。

        
            优先级：Session 设置 > Env 设置


            禁用每个 session 自己的线程池，便于统一线程管理 ，这个开了下面这个没用了
            session_options.DisablePerSessionThreads();

            我现在这个强化学期 ， mlp 模型 ， 只需要用一个 session ， 统一管理就行了 ，不用session_options 
        */
        session_options.SetIntraOpNumThreads(1);
        // 开启较高等级的图优化。 ONNX Runtime 可能会做算子融合、常量折叠等优化，通常能提升推理效率。
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);


        /*
            Session 是“加载好的模型实例”。
            只要模型路径正确，后面每来一条 IMU 数据，都可以重复复用同一个 Session 推理，不需要每次循环里重新加载 onnx 文件。
        */

        Ort::Session session(env, model_path.c_str(), session_options);

        // 分配器主要用于安全地获取模型中的输入输出名称等元信息。
        Ort::AllocatorWithDefaultOptions allocator;

        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            std::cerr << "Expected 1 input and 1 output, but got input_count="
                      << session.GetInputCount() << ", output_count=" << session.GetOutputCount()
                      << std::endl;
            return 1;
        }


        /*
        
        从模型图中读取输入名和输出名。
        这一步非常关键，因为 session.Run(...) 不是按“位置”盲传，
        而是按“节点名字”绑定输入输出。
        
        虽然我们训练导出时已经指定过：
        input_names=["imu_input"]
        output_names=["change_logit"]
        但在 C++ 部署里，动态读取名字更稳妥，避免手写字符串和模型不一致。


        多个input 名字的场景：
        动态的获得了输入模型的  name 节点 和 shape  

            // 获取模型输入数量，并为输入名字和输入缓冲区分配空间。
            ctx->num_inputs = ctx->session->GetInputCount();
            ctx->input_names.resize(ctx->num_inputs);
            ctx->input_buffer.resize(input_size);


            // 这里有问题

            // 读取每个输入节点的名字和 shape 信息。
            for (size_t i = 0; i < ctx->num_inputs; i++) {
                Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
                ctx->input_names[i] = input_name.get();

                // 这里会被覆盖，有问题
                auto type_info = ctx->session->GetInputTypeInfo(i);
                ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();

                // 如果 batch 维是动态维度 -1，这里默认按 batch=1 处理。
                if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
            }



            mlp 多层状态感知机 ， 导出onnx 只有一个 节点
            基础的全连接网络（MLP）：通常只有一个输入节点，你的代码逻辑基本适用。



            size_t input_count = session.GetInputCount();
            std::vector<std::string> input_name_store;
            std::vector<const char*> input_names;

            for (size_t i = 0; i < input_count; ++i) {
                auto name = session.GetInputNameAllocated(i, allocator);
                
                input_name_store.emplace_back(name.get());
                input_names.push_back(input_name_store.back().c_str());
            }
                    
        */


        auto input_name_holder = session.GetInputNameAllocated(0, allocator);
        auto output_name_holder = session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_holder.get();
        const char* output_name = output_name_holder.get();

        // 先读掉 csv 表头。
        std::string line;
        if (!std::getline(input_file, line)) {
            std::cerr << "CSV file is empty: " << csv_path << std::endl;
            return 1;
        }

        std::cout << "Model: " << model_path << std::endl;
        std::cout << "CSV: " << csv_path << std::endl;
        std::cout << "Streaming IMU samples and judging in real time..." << std::endl;

        bool has_previous_timestamp = false;
        int64_t previous_timestamp_ms = 0;
        int sample_index = 0;
        int correct_count = 0;
        int total_count = 0;

        // 主框架 3：
        // 逐行读取 IMU 数据，并按时间戳做“实时”推理。
        while (std::getline(input_file, line)) {
            if (line.empty()) {
                continue;
            }

            ImuSample sample;
            if (!ParseSample(line, sample)) {
                std::cerr << "Skip malformed line: " << line << std::endl;
                continue;
            }

            // 主框架 4：
            // 用相邻两条数据的时间戳差值来模拟实时输入节奏。
            // 例如上一条是 100ms，这一条是 120ms，就等待 20ms 再推理。
            if (has_previous_timestamp) {
                const int64_t delta_ms = sample.timestamp_ms - previous_timestamp_ms;
                if (delta_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(delta_ms));
                }
            }
            previous_timestamp_ms = sample.timestamp_ms;
            has_previous_timestamp = true;

            // 主框架 5：
            // 对当前这一帧 IMU 数据做一次推理。
            // 这一步内部会完成：
            // 1. 构造 [1, 6] 输入 Tensor
            // 2. 调用 ONNX Runtime 执行前向计算
            // 3. 取回模型输出的原始 logit
            const float logit =
                RunSingleInference(session, input_name, output_name, sample.features);

            // 主框架 6：
            // 把 logit 转成概率，再按照 0.5 阈值判断是否剧烈变化。
            const float probability = Sigmoid(logit);
            const int predicted_label = probability >= 0.5f ? 1 : 0;

            ++sample_index;
            ++total_count;
            if (sample.label == predicted_label) {
                ++correct_count;
            }

            std::cout << "[sample " << sample_index << "] "
                      << "t=" << sample.timestamp_ms << "ms "
                      << "roll=" << sample.features[0] << ", pitch=" << sample.features[1]
                      << ", yaw=" << sample.features[2] << ", gx=" << sample.features[3]
                      << ", gy=" << sample.features[4] << ", gz=" << sample.features[5]
                      << " => logit=" << std::fixed << std::setprecision(4) << logit
                      << ", prob=" << probability
                      << ", pred=" << predicted_label;

            if (sample.label >= 0) {
                std::cout << ", label=" << sample.label;
            }
            std::cout << std::endl;
        }

        // 主框架 7：
        // 如果 csv 中有真实标签，就顺手统计整体准确率，方便验证模型效果。
        if (total_count == 0) {
            std::cerr << "No valid IMU samples found in: " << csv_path << std::endl;
            return 1;
        }

        std::cout << "Finished. total_samples=" << total_count
                  << ", accuracy=" << std::fixed << std::setprecision(4)
                  << (static_cast<float>(correct_count) / static_cast<float>(total_count))
                  << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
