/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// 头文件保护宏：首次包含时 DATA_UTILS_H 尚未定义，因此继续展开本文件内容。
#ifndef DATA_UTILS_H
// 定义头文件保护宏，避免同一翻译单元重复包含本文件后出现函数或宏的重复定义。
#define DATA_UTILS_H

// 引入 POSIX 文件控制接口，提供 open 以及 O_RDWR、O_CREAT、O_TRUNC 等标志。
#include <fcntl.h>
// 引入 POSIX 文件权限定义，提供 S_IRUSR、S_IWUSR 等 mode 位。
#include <sys/stat.h>
// 引入 POSIX 基础 I/O 接口，提供 write 和 close。
#include <unistd.h>
// 引入 C++ 文件流，ReadFile 使用 std::ifstream 以二进制方式读取输入。
#include <fstream>

// 统一错误日志格式：向 stdout 输出错误级别、格式化正文和换行；##args 允许调用时没有可变参数。
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [in] bufferSize: expected file size
 * @param [out] buffer: buffer to store data
 * @param [in] bufferLen: buffer length
 * @return read result
 */
// 定义内联读取函数：要求文件大小恰好等于 bufferSize，并把全部二进制内容写入调用方缓冲区。
inline bool ReadFile(const std::string &filePath, size_t bufferSize, void *buffer, size_t bufferLen)
// 进入 ReadFile 函数体；函数任一校验或 I/O 步骤失败时立即返回 false。
{
    // 首先验证目标地址有效，防止后续 file.read 向空指针写入而触发未定义行为。
    if (buffer == nullptr) {
        // 记录空缓冲区这一调用方错误，便于定位 Host 内存分配或参数传递问题。
        ERROR_LOG("buffer is nullptr");
        // 终止本次读取，并用 false 告知调用方没有获得有效输入数据。
        return false;
    // 结束空指针校验分支；非空缓冲区继续检查容量。
    }
    // 验证预期读取字节数没有超过实际缓冲区容量，避免发生越界写。
    if (bufferSize > bufferLen) {
        // 输出预期数据大于缓冲区的错误原因；这里不进行截断读取，以免悄悄产生残缺张量。
        ERROR_LOG("buffer size is larger than buffer length");
        // 返回失败，保持“成功即完整读取”的函数契约。
        return false;
    // 结束缓冲区容量不足分支。
    }

    // 以二进制只读方式打开输入文件，避免文本模式对字节（尤其换行）进行平台相关转换。
    std::ifstream file(filePath, std::ios::binary);
    // 检查文件流是否成功关联到目标文件，例如路径不存在或权限不足都会进入该分支。
    if (!file.is_open()) {
        // 输出无法打开的具体路径；c_str() 将 std::string 转为 printf 风格的 C 字符串。
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        // 文件未打开，直接返回失败，不能继续执行定位或读取。
        return false;
    // 结束文件打开失败分支。
    }

    // 将读位置移动到文件末尾，为下一行通过当前位置计算文件总字节数做准备。
    file.seekg(0, std::ios::end);
    // 读取末尾位置作为文件大小；二进制文件中该位置对应从文件头起的字节偏移。
    size_t fileSize = file.tellg();
    // 将读位置恢复到文件开头，否则后续 read 会从末尾开始而读不到数据。
    file.seekg(0, std::ios::beg);

    // 要求磁盘文件大小与张量所需字节数完全一致，防止形状或 dtype 不匹配被静默接受。
    if (fileSize != bufferSize) {
        // 同时打印实际大小和期望大小，便于判断输入生成脚本是否使用了错误形状或数据类型。
        ERROR_LOG("file size %zu != expected size %zu", fileSize, bufferSize);
        // 主动关闭已打开的文件流；即使析构也会关闭，这里使失败路径的资源释放更明确。
        file.close();
        // 返回失败，不把尺寸不一致的文件内容传入设备计算。
        return false;
    // 结束文件大小不匹配分支。
    }

    // 把 void* 转为字符指针并精确读取 bufferSize 字节，因为流的无格式 read 接口以 char* 表示原始字节缓冲区。
    file.read(static_cast<char *>(buffer), bufferSize);
    // 检查读取后的流状态；短读、介质错误等都会让流转换为 false。
    if (!file) {
        // 记录完整读取失败；此时缓冲区可能仅含部分数据，调用方不应继续使用。
        ERROR_LOG("Read file failed");
        // 关闭文件描述资源，确保错误退出不依赖稍后的局部对象析构。
        file.close();
        // 返回失败，表明缓冲区内容不满足完整输入契约。
        return false;
    // 结束读取错误处理分支。
    }
    
    // 完整读取成功后关闭文件，尽早释放底层文件描述符。
    file.close();
    // 返回 true，表示路径、尺寸和实际读取三个步骤均成功。
    return true;
// 结束 ReadFile 函数定义。
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
// 定义内联写文件函数：把 Host 缓冲区中的 size 字节输出为一个全新的二进制结果文件。
inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
// 进入 WriteFile 函数体；所有资源均在返回前显式释放。
{
    // 校验源缓冲区非空，防止 POSIX write 从无效地址读取数据。
    if (buffer == nullptr) {
        // 记录失败原因，提示问题来自调用方传入的输出 Host 缓冲区。
        ERROR_LOG("Write file failed. buffer is nullptr");
        // 空指针情况下不创建或修改输出文件，直接报告失败。
        return false;
    // 结束空指针检查分支。
    }

    // 以读写、必要时创建、已存在则截断的方式打开文件，并把所有者权限设为可读可写。
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    // POSIX open 返回负值表示打开失败，常见原因包括目录不存在或无写权限。
    if (fd < 0) {
        // 输出目标路径，帮助定位输出目录或权限配置错误。
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        // 未取得有效文件描述符，因此立即返回失败。
        return false;
    // 结束文件打开失败分支。
    }

    // 请求一次性把 size 字节写入文件，并保存系统调用实际写入的字节数。
    ssize_t writeSize = write(fd, buffer, size);
    // 无论写入是否完整都关闭文件描述符，避免 Host 侧文件资源泄漏。
    close(fd);
    // 将 write 的有符号返回值转为 size_t 后与期望字节数比较，以识别错误返回或短写。
    if (static_cast<size_t>(writeSize) != size) {
        // 记录写入不完整；函数不尝试循环补写，因此任何短写都视为失败。
        ERROR_LOG("Write file Failed.");
        // 返回失败，提示输出文件不能作为可信的完整结果使用。
        return false;
    // 结束短写或写入错误处理分支。
    }

    // 返回 true，表示目标文件已成功打开且一次 write 写满全部请求字节。
    return true;
// 结束 WriteFile 函数定义。
}

// 结束 DATA_UTILS_H 头文件保护条件，使本文件可被安全地多次 include。
#endif
