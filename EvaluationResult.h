#pragma once

#include <cmath>
#include <limits>

enum class EvalMode
{
    Classification,
    Regression
};

struct EvaluationResult
{
    EvalMode mode = EvalMode::Classification;

    // Classification (default to worst possible: -1)
    int correct = -1;
    int total   = 0;

    // Regression (default to worst possible: infinity)
    double mse  = std::numeric_limits<double>::infinity();
    double mae  = std::numeric_limits<double>::infinity();

    // Checkpointing
    int epoch = 0;

    // Comparison operators for checkpointing
    // Returns true if this evaluation is better than other
    // For regression: lower MAE is better (primary), lower MSE is better (secondary)
    // For classification: higher accuracy is better
    bool operator<(const EvaluationResult& other) const
    {
        if (mode == EvalMode::Regression)
        {
            if (std::abs(mae - other.mae) > 1e-10)
            {
                return mae < other.mae; // lower MAE is better
            }
            return mse < other.mse; // if MAE equal, lower MSE is better
        }
        else // Classification
        {
            return correct > other.correct; // higher accuracy is better
        }
    }

    bool operator>(const EvaluationResult& other) const
    {
        return other < *this;
    }
};
