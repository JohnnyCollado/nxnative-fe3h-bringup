// Compatibility layer (dm/third_party/lsfg-eden/README.md): the part of Eden's Common::FS the frame generation sources use, on stdio.
// Write / Read of an object return 1 when it went through; of a span, vector or string, the number of elements that went through.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>
#include "common/common_types.h"
namespace Common::FS {
enum class FileAccessMode { Read, Write };
enum class FileType { BinaryFile, TextFile };
class IOFile {
public:
    IOFile(const std::filesystem::path& path, FileAccessMode mode, FileType = FileType::BinaryFile) {
        f = std::fopen(path.string().c_str(), mode == FileAccessMode::Read ? "rb" : "wb");
    }
    ~IOFile() { if (f) std::fclose(f); }
    IOFile(const IOFile&) = delete;
    IOFile& operator=(const IOFile&) = delete;
    [[nodiscard]] bool IsOpen() const { return f != nullptr; }
    [[nodiscard]] u64 GetSize() const {
        if (!f) return 0;
        const long at = std::ftell(f); std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, at, SEEK_SET);
        return n < 0 ? 0 : static_cast<u64>(n);
    }
    bool Flush() { return f && std::fflush(f) == 0; }
    std::size_t Write(const std::string& s) { return f ? std::fwrite(s.data(), 1, s.size(), f) : 0; }
    template <typename T> std::size_t Write(std::span<const T> s) { return f ? std::fwrite(s.data(), sizeof(T), s.size(), f) : 0; }
    template <typename T> std::size_t Write(std::span<T> s) { return f ? std::fwrite(s.data(), sizeof(T), s.size(), f) : 0; }
    template <typename T> std::size_t Write(const std::vector<T>& v) { return f ? std::fwrite(v.data(), sizeof(T), v.size(), f) : 0; }
    template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>> std::size_t Write(const T& v) { return f ? std::fwrite(&v, sizeof(T), 1, f) : 0; }
    template <typename T> std::size_t Read(std::vector<T>& v) { return f ? std::fread(v.data(), sizeof(T), v.size(), f) : 0; }
    template <typename T> std::size_t Read(std::span<T> s) { return f ? std::fread(s.data(), sizeof(T), s.size(), f) : 0; }
    template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>> std::size_t Read(T& v) { return f ? std::fread(&v, sizeof(T), 1, f) : 0; }
private:
    std::FILE* f{};
};
inline bool Exists(const std::filesystem::path& p) { std::error_code e; return std::filesystem::exists(p, e); }
inline bool RemoveFile(const std::filesystem::path& p) { std::error_code e; return std::filesystem::remove(p, e); }
inline bool CreateDirs(const std::filesystem::path& p) { std::error_code e; std::filesystem::create_directories(p, e); return std::filesystem::is_directory(p, e); }
} // namespace Common::FS
