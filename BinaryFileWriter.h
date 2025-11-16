#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

class BinaryFileWriter {
public:
    explicit BinaryFileWriter(const std::string& path)
            : f_(path, std::ios::binary)
    {
        if (!f_) throw std::runtime_error("Cannot open file for writing: " + path);
    }

    // Writes a 4-character code (e.g. "EXDS", "VALD")
    void write_tag(const char tag[4]) {
        f_.write(tag, 4);
    }

    // Writes a 32-bit number in big-endian format.
    // Big-endian means the most significant byte is written first.
    void write_be_u32(uint32_t x) {
        uint8_t b[4] = {
                static_cast<uint8_t>((x >> 24) & 0xFF),
                static_cast<uint8_t>((x >> 16) & 0xFF),
                static_cast<uint8_t>((x >> 8)  & 0xFF),
                static_cast<uint8_t>(x & 0xFF)
        };
        f_.write(reinterpret_cast<const char*>(b), 4);
    }

    // Writes a floating-point number directly (keeps full numeric precision).
    void write_double(double x) {
        f_.write(reinterpret_cast<const char*>(&x), sizeof(double));
    }

private:
    std::ofstream f_;
};
