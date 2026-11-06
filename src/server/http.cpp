#include "server/http.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace tsrv {
namespace {

constexpr size_t kMaxRequest = 1u << 20;  // 1 MiB is plenty for this API

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

size_t headerEnd(const std::string& buf) {
  size_t p = buf.find("\r\n\r\n");
  if (p != std::string::npos) return p + 4;
  p = buf.find("\n\n");
  return p == std::string::npos ? std::string::npos : p + 2;
}

long long contentLength(const std::map<std::string, std::string>& headers) {
  auto it = headers.find("content-length");
  if (it == headers.end()) return 0;
  char* end = nullptr;
  long long n = std::strtoll(it->second.c_str(), &end, 10);
  return (n > 0 && n < static_cast<long long>(kMaxRequest)) ? n : 0;
}

bool readRequest(socket_t fd, HttpRequest& req) {
  std::string buf;
  char chunk[4096];

  // Phase 1: full header block.
  size_t headEnd = headerEnd(buf);
  while (headEnd == std::string::npos) {
    int n = ::recv(fd, chunk, static_cast<int>(sizeof(chunk)), 0);
    if (n <= 0) return false;
    buf.append(chunk, static_cast<size_t>(n));
    if (buf.size() > kMaxRequest) return false;
    headEnd = headerEnd(buf);
  }

  std::vector<std::string> lines;
  {
    std::string head = buf.substr(0, headEnd);
    size_t pos = 0;
    while (pos < head.size()) {
      size_t eol = head.find('\n', pos);
      if (eol == std::string::npos) eol = head.size();
      std::string line = head.substr(pos, eol - pos);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
      pos = eol + 1;
    }
  }
  if (!lines.empty()) {
    size_t sp1 = lines[0].find(' ');
    size_t sp2 = sp1 == std::string::npos ? std::string::npos
                                          : lines[0].find(' ', sp1 + 1);
    if (sp1 != std::string::npos && sp2 != std::string::npos) {
      req.method = lines[0].substr(0, sp1);
      std::string target = lines[0].substr(sp1 + 1, sp2 - sp1 - 1);
      size_t q = target.find('?');
      req.path = urlDecode(q == std::string::npos ? target : target.substr(0, q));
      if (q != std::string::npos) req.query = target.substr(q + 1);
    }
  }
  for (size_t i = 1; i < lines.size(); ++i) {
    size_t colon = lines[i].find(':');
    if (colon == std::string::npos) continue;
    req.headers[lower(lines[i].substr(0, colon))] =
        trim(lines[i].substr(colon + 1));
  }

  // Phase 2: body by Content-Length.
  const size_t len = static_cast<size_t>(contentLength(req.headers));
  while (buf.size() < headEnd + len) {
    int n = ::recv(fd, chunk, static_cast<int>(sizeof(chunk)), 0);
    if (n <= 0) return false;
    buf.append(chunk, static_cast<size_t>(n));
    if (buf.size() > kMaxRequest) return false;
  }
  req.body = buf.substr(headEnd, len);
  return !req.method.empty();
}

bool writeAll(socket_t fd, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
    int n = ::send(fd, data.data() + sent,
                   static_cast<int>(std::min<size_t>(data.size() - sent, 1 << 16)), 0);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

void closeSocket(socket_t fd) {
#ifdef _WIN32
  ::closesocket(fd);
#else
  ::close(fd);
#endif
}

}  // namespace

std::string urlDecode(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      out += ' ';
    } else if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      int a = hex(s[i + 1]), b = hex(s[i + 2]);
      if (a >= 0 && b >= 0) {
        out += static_cast<char>((a << 4) | b);
        i += 2;
      } else {
        out += s[i];
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

std::string HttpRequest::queryValue(const std::string& key) const {
  size_t pos = 0;
  while (pos < query.size()) {
    size_t amp = query.find('&', pos);
    if (amp == std::string::npos) amp = query.size();
    std::string part = query.substr(pos, amp - pos);
    size_t eq = part.find('=');
    if (eq != std::string::npos && urlDecode(part.substr(0, eq)) == key) {
      return urlDecode(part.substr(eq + 1));
    }
    pos = amp + 1;
  }
  return {};
}

std::string reasonPhrase(int status) {
  switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    default: return "OK";
  }
}

HttpServer::~HttpServer() {
#ifdef _WIN32
  if (listenFd_ != INVALID_SOCKET) closeSocket(listenFd_);
  if (wsaStarted_) ::WSACleanup();
#else
  if (listenFd_ >= 0) closeSocket(listenFd_);
#endif
}

bool HttpServer::start(int port) {
#ifdef _WIN32
  WSADATA wsa;
  if (::WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
    std::cerr << "WSAStartup failed\n";
    return false;
  }
  wsaStarted_ = true;
#endif
  listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
  if (listenFd_ == INVALID_SOCKET) return false;
#else
  if (listenFd_ < 0) return false;
#endif
  int reuse = 1;
  ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<unsigned short>(port));
  if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::cerr << "bind failed on port " << port << "\n";
    return false;
  }
  if (::listen(listenFd_, 64) != 0) {
    std::cerr << "listen failed\n";
    return false;
  }
  port_ = port;
  return true;
}

void HttpServer::serve(const HttpHandler& handler) {
  running_ = true;
  while (running_) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(listenFd_, &fds);
    timeval tv{0, 200000};  // 200 ms so stop() is responsive
    int ready = ::select(static_cast<int>(listenFd_) + 1, &fds, nullptr, nullptr, &tv);
    if (ready <= 0) continue;
    socket_t fd = ::accept(listenFd_, nullptr, nullptr);
#ifdef _WIN32
    if (fd == INVALID_SOCKET) continue;
#else
    if (fd < 0) continue;
#endif
#ifdef _WIN32
    DWORD timeoutMs = 3000;
#else
    timeval timeout{3, 0};
#endif
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
#ifdef _WIN32
                 reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
#else
                 reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#endif

    HttpRequest req;
    HttpResponse resp;
    if (!readRequest(fd, req)) {
      resp.status = 400;
      resp.body = "bad request";
      resp.contentType = "text/plain; charset=utf-8";
    } else {
      try {
        resp = handler(req);
      } catch (const std::exception& e) {
        std::cerr << "handler error: " << e.what() << "\n";
        resp = HttpResponse{};
        resp.status = 500;
        resp.body = "{\"error\":\"internal server error\"}";
      } catch (...) {
        resp = HttpResponse{};
        resp.status = 500;
        resp.body = "{\"error\":\"internal server error\"}";
      }
    }

    std::string out;
    out += "HTTP/1.1 " + std::to_string(resp.status) + " " +
           reasonPhrase(resp.status) + "\r\n";
    out += "Content-Type: " + resp.contentType + "\r\n";
    out += "Content-Length: " + std::to_string(resp.body.size()) + "\r\n";
    out += "Access-Control-Allow-Origin: *\r\n";
    out += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    out += "Access-Control-Allow-Headers: Content-Type, X-Session-Id\r\n";
    out += "Cache-Control: no-store\r\n";
    out += "Connection: close\r\n";
    for (const auto& kv : resp.extraHeaders) {
      out += kv.first + ": " + kv.second + "\r\n";
    }
    out += "\r\n";
    out += resp.body;
    writeAll(fd, out);
    closeSocket(fd);
  }
}

void HttpServer::stop() { running_ = false; }

}  // namespace tsrv
