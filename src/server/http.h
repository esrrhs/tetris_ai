// Zero-dependency blocking HTTP/1.1 server (single-threaded event loop).
// Mirrors the request/response shape of doudizhu_ai/server/lua/http_server.lua:
// one request per connection, CORS on every response, X-Session-Id isolation.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
using socket_t = SOCKET;
#else
using socket_t = int;
#endif

namespace tsrv {

struct HttpRequest {
  std::string method;
  std::string path;   // percent-decoded, without query
  std::string query;  // raw query string
  std::string body;
  std::map<std::string, std::string> headers;  // lower-case keys

  std::string header(const std::string& key) const {
    auto it = headers.find(key);
    return it == headers.end() ? std::string{} : it->second;
  }
  std::string queryValue(const std::string& key) const;
};

struct HttpResponse {
  int status = 200;
  std::string contentType = "application/json; charset=utf-8";
  std::string body;
  std::vector<std::pair<std::string, std::string>> extraHeaders;
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
 public:
  ~HttpServer();
  bool start(int port);
  void serve(const HttpHandler& handler);  // blocks until stop()
  void stop();

  int port() const { return port_; }

 private:
#ifdef _WIN32
  socket_t listenFd_ = INVALID_SOCKET;
#else
  socket_t listenFd_ = -1;
#endif
  int port_ = 0;
  bool running_ = false;
  bool wsaStarted_ = false;
};

std::string urlDecode(const std::string& s);
std::string reasonPhrase(int status);

}  // namespace tsrv
