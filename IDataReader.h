
#pragma once

#include "Network.h"
#include <fstream>

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
    virtual const std::vector<std::pair<VectorXd,int>>& test() const = 0;
};
