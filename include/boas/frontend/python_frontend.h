#ifndef BOAS_FRONTEND_PYTHON_FRONTEND_H
#define BOAS_FRONTEND_PYTHON_FRONTEND_H

#include <memory>
#include <string>
#include "boas/ast/ast.h"

namespace boas {

class PythonFrontendImpl;

class PythonFrontend {
public:
    PythonFrontend();
    ~PythonFrontend();

    std::unique_ptr<AST> parse(const std::string& source);

private:
    std::unique_ptr<PythonFrontendImpl> impl_;
};

} // namespace boas

#endif // BOAS_FRONTEND_PYTHON_FRONTEND_H
