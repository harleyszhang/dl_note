#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

class fileWrapper {
public:
    // 构造函数 打开文件
    explicit fileWrapper(const std::string& filename) : 
    m_file(filename, std::ios::in | std::ios::out | std::ios::app) {
        if (!m_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    // 析构函数 关闭文件
    ~fileWrapper() {
        if (m_file.is_open()) {
            throw std::runtime_error("File is need to close");
            m_file.close();
        }
    }

    void writeLine(const std::string& line) {
        if (!m_file.is_open()) {
            throw std::runtime_error("Failed to write file, file is close.");
        }
        m_file << line << std::endl;
    }

private:
    std::fstream m_file;
};

template <typename T>
void wrapper(T&& val) {
    // 完美转发
    std::cout << "input val is " << std::forward<T>(val) << std::endl;
}

int main() {
    
    int x = 11;
    wrapper(67);
    wrapper("hello world");
    wrapper(x);

    try {
        fileWrapper fw("test.txt");
        fw.writeLine("Hello World");
    } 
    // 离开作用域自动调用 ~FileWrapper()，关闭文件
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

