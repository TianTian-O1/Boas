#pragma once
#include <string>
#include <variant>

namespace boas {

class Value {
public:
  Value() : value_(0.0) {}
  Value(double d) : value_(d) {}
  Value(const std::string& s) : value_(s) {}
  Value(bool b) : value_(b) {}

  bool isString() const { return std::holds_alternative<std::string>(value_); }
  bool isNumber() const { return std::holds_alternative<double>(value_); }
  bool isBoolean() const { return std::holds_alternative<bool>(value_); }

  double asNumber() const { return std::get<double>(value_); }
  std::string asString() const { return std::get<std::string>(value_); }
  bool asBoolean() const { return std::get<bool>(value_); }

private:
  std::variant<double, std::string, bool> value_;
};

} // namespace boas