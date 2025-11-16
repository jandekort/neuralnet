
#pragma once

#include "Network.h"
#include <fstream>

static uint32_t be_u32(std::ifstream& f)
{
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4);
    return (uint32_t(b[0])<<24) | (uint32_t(b[1])<<16) | (uint32_t(b[2])<<8) | uint32_t(b[3]);
}

// ============================================================================
// Data reading interfaces
// ============================================================================
class IDataReader
{
public:
    using Example = Network::Example;
    virtual ~IDataReader() = default;
    virtual void load() = 0;
    virtual const std::vector<Example>& training() const = 0;
    virtual const std::vector<Example>& test() const = 0;
};
