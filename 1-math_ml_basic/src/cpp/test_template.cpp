#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cassert>

using namespace std;

// 通过枚举类定义设备/后端类型
enum class Device {
    CPU,
    GPU,
    NPU
};

// 1. 先声明一个通用的模板类 Tensor，形参包括 “后端（Device）” 和 “数据类型（typename T）”
template<Device Dev, typename T>
class Tensor;

// 2. 通过偏特化：Tensor<CPU, T>，实现 CPU 后端的 Tensor 类设计
/*特例化哪个参数，哪个参数就不用定义在 template<> 中*/
template<typename T>
class Tensor<Device::CPU, T> {
public:
    // 显式构造函数: 传入 vector 类型张量形状参数
    explicit Tensor(const vector<size_t> shape): shape(shape) {
        size = 1;
        for (auto dim_size:shape) {
            size *= dim_size;
        }
        data.resize(size);
    }
    
    // 简单的张量索引访问接口
    const T& operator[](size_t index) const {
        return data[index];
    }
    // const 表示这是一个常量成员函数
    size_t size() const noexcept {
        return size;
    }
    const std::vector<size_t>& shape() const { return shape; }

    // 张量的一些成员函数
    void add(const Tensor<Device::CPU, T> input_t) {
        assert size == input_t.size();
        for (int i=0; i < size; i++) {
            data[i] += input_t.data[i];
        }
    }
    
    // Debug: 打印部分数据
    void printDebug(const std::string& name) const {
        std::cout << "[CPU Tensor<" << typeid(T).name() << ">] " 
                  << name << " shape: [";
        for (auto s : shape) std::cout << s << " ";
        std::cout << "], data[0] = " << data[0] << " ...\n";
    }

private:
    size_t size;
    std::vector<size_t> shape;
    std::vector<size_t> data;
};

template<typename T1,typename T2>
auto func(const T1& x, const T2& y) {
    return x + y;
}

// 模板全特化
template<>
auto func(const int& x, const double& y) {
    return x-y;
}

int main() {
    auto result1 = add(3, 5);
    auto result2 = add(8.7, 9.0);
    std::cout << "result1: " << result1 << std::endl;
    std::cout << "result2: " << result2 << std::endl;
}
