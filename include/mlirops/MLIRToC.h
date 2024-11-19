// include/mlirops/MLIRToC.h
#pragma once
#include <string>
#include <vector>
#include <memory>

namespace matrix {

struct MatrixInfo {
    int rows;
    int cols;
    std::vector<double> values;
    std::string name;
};

class MLIRToC {
public:
    static std::string convertToC(const std::string& mlirInput);

private:
    static std::vector<MatrixInfo> parseMatrices(const std::string& mlirInput);
    static std::string generateMatrixCode(const std::vector<MatrixInfo>& matrices);
    static std::string generateMatrixOps(const std::vector<MatrixInfo>& matrices);
};

} // namespace matrix