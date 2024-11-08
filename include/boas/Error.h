#pragma once
#include <string>

namespace boas {

class Error {
public:
  explicit Error(const std::string& message) : message_(message) {}
  const std::string& getMessage() const { return message_; }
private:
  std::string message_;
};

} // namespace boas