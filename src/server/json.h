// Minimal JSON value + parser/serializer for the web service.
// Header-only on purpose: the server must build with nothing but a C++20
// compiler (no third-party HTTP/JSON dependency, no Lua runtime).
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace tsrv {

class Json {
 public:
  enum class Type { Null, Bool, Int, Real, String, Array, Object };

  Json() = default;

  static Json null() { return Json{}; }
  static Json boolean(bool v) {
    Json j;
    j.t_ = Type::Bool;
    j.b_ = v;
    return j;
  }
  static Json integer(long long v) {
    Json j;
    j.t_ = Type::Int;
    j.i_ = v;
    return j;
  }
  static Json real(double v) {
    Json j;
    j.t_ = Type::Real;
    j.d_ = v;
    return j;
  }
  static Json str(std::string v) {
    Json j;
    j.t_ = Type::String;
    j.s_ = std::move(v);
    return j;
  }
  static Json array() {
    Json j;
    j.t_ = Type::Array;
    return j;
  }
  static Json object() {
    Json j;
    j.t_ = Type::Object;
    return j;
  }

  Type type() const { return t_; }
  bool isNull() const { return t_ == Type::Null; }
  bool isArray() const { return t_ == Type::Array; }
  bool isObject() const { return t_ == Type::Object; }

  Json& set(const std::string& key, Json value) {
    if (t_ != Type::Object) {
      t_ = Type::Object;
      obj_.clear();
    }
    for (auto& kv : obj_) {
      if (kv.first == key) {
        kv.second = std::move(value);
        return *this;
      }
    }
    obj_.emplace_back(key, std::move(value));
    return *this;
  }

  Json& push(Json value) {
    if (t_ != Type::Array) {
      t_ = Type::Array;
      arr_.clear();
    }
    arr_.push_back(std::move(value));
    return *this;
  }

  const Json* find(const std::string& key) const {
    if (t_ != Type::Object) return nullptr;
    for (const auto& kv : obj_) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

  const std::vector<Json>& items() const { return arr_; }
  const std::vector<std::pair<std::string, Json>>& members() const { return obj_; }

  std::string asString(const std::string& fallback = "") const {
    return t_ == Type::String ? s_ : fallback;
  }
  long long asInt(long long fallback = 0) const {
    if (t_ == Type::Int) return i_;
    if (t_ == Type::Real) return static_cast<long long>(d_);
    if (t_ == Type::Bool) return b_ ? 1 : 0;
    if (t_ == Type::String) {
      char* end = nullptr;
      long long v = std::strtoll(s_.c_str(), &end, 10);
      return end == s_.c_str() ? fallback : v;
    }
    return fallback;
  }
  double asReal(double fallback = 0.0) const {
    if (t_ == Type::Real) return d_;
    if (t_ == Type::Int) return static_cast<double>(i_);
    if (t_ == Type::String) return std::strtod(s_.c_str(), nullptr);
    return fallback;
  }
  bool asBool(bool fallback = false) const {
    if (t_ == Type::Bool) return b_;
    if (t_ == Type::Int) return i_ != 0;
    if (t_ == Type::String) return s_ == "true" || s_ == "1";
    return fallback;
  }

  std::string dump() const {
    std::string out;
    write(out);
    return out;
  }

  static Json parse(const std::string& text) {
    Parser p{text, 0};
    Json v = p.value();
    return v;
  }

 private:
  void write(std::string& out) const {
    switch (t_) {
      case Type::Null:
        out += "null";
        break;
      case Type::Bool:
        out += b_ ? "true" : "false";
        break;
      case Type::Int:
        out += std::to_string(i_);
        break;
      case Type::Real: {
        if (!std::isfinite(d_)) {
          out += "0";
          break;
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", d_);
        out += buf;
        break;
      }
      case Type::String:
        writeString(out, s_);
        break;
      case Type::Array: {
        out += '[';
        for (size_t i = 0; i < arr_.size(); ++i) {
          if (i) out += ',';
          arr_[i].write(out);
        }
        out += ']';
        break;
      }
      case Type::Object: {
        out += '{';
        for (size_t i = 0; i < obj_.size(); ++i) {
          if (i) out += ',';
          writeString(out, obj_[i].first);
          out += ':';
          obj_[i].second.write(out);
        }
        out += '}';
        break;
      }
    }
  }

  static void writeString(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
          } else {
            out += static_cast<char>(c);
          }
      }
    }
    out += '"';
  }

  struct Parser {
    const std::string& s;
    size_t i;

    void ws() {
      while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool eat(char c) {
      ws();
      if (i < s.size() && s[i] == c) {
        ++i;
        return true;
      }
      return false;
    }
    Json value() {
      ws();
      if (i >= s.size()) return Json::null();
      char c = s[i];
      if (c == '{') return objectValue();
      if (c == '[') return arrayValue();
      if (c == '"') return Json::str(stringValue());
      if (c == 't' || c == 'f' || c == 'n') {
        if (s.compare(i, 4, "true") == 0) { i += 4; return Json::boolean(true); }
        if (s.compare(i, 5, "false") == 0) { i += 5; return Json::boolean(false); }
        if (s.compare(i, 4, "null") == 0) { i += 4; return Json::null(); }
        ++i;
        return Json::null();
      }
      return numberValue();
    }
    Json objectValue() {
      Json o = Json::object();
      ++i;  // {
      ws();
      if (i < s.size() && s[i] == '}') { ++i; return o; }
      while (i < s.size()) {
        ws();
        if (i >= s.size() || s[i] != '"') break;
        std::string key = stringValue();
        if (!eat(':')) break;
        o.set(key, value());
        ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') ++i;
        break;
      }
      return o;
    }
    Json arrayValue() {
      Json a = Json::array();
      ++i;  // [
      ws();
      if (i < s.size() && s[i] == ']') { ++i; return a; }
      while (i < s.size()) {
        a.push(value());
        ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == ']') ++i;
        break;
      }
      return a;
    }
    std::string stringValue() {
      std::string out;
      ++i;  // opening quote
      while (i < s.size()) {
        char c = s[i++];
        if (c == '"') break;
        if (c != '\\') { out += c; continue; }
        if (i >= s.size()) break;
        char e = s[i++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            if (i + 4 > s.size()) break;
            unsigned cp = 0;
            for (int k = 0; k < 4; ++k) {
              char h = s[i++];
              cp <<= 4;
              if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
            }
            // BMP → UTF-8; non-BMP escapes are not needed by this API.
            if (cp < 0x80) {
              out += static_cast<char>(cp);
            } else if (cp < 0x800) {
              out += static_cast<char>(0xC0 | (cp >> 6));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (cp >> 12));
              out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            break;
          }
          default: out += e; break;
        }
      }
      return out;
    }
    Json numberValue() {
      size_t start = i;
      while (i < s.size() &&
             (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' ||
              s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) {
        ++i;
      }
      if (start == i) { ++i; return Json::null(); }
      std::string tok = s.substr(start, i - start);
      if (tok.find_first_of(".eE") != std::string::npos) {
        return Json::real(std::strtod(tok.c_str(), nullptr));
      }
      return Json::integer(std::strtoll(tok.c_str(), nullptr, 10));
    }
  };

  Type t_ = Type::Null;
  bool b_ = false;
  long long i_ = 0;
  double d_ = 0.0;
  std::string s_;
  std::vector<Json> arr_;
  std::vector<std::pair<std::string, Json>> obj_;
};

}  // namespace tsrv
