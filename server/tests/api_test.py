#!/usr/bin/env python3
"""HTTP contract regression for tetris_ai_server.

    python server/tests/api_test.py
    python server/tests/api_test.py --base-url http://127.0.0.1:8080
    python server/tests/api_test.py --exe build_msvc/tetris_ai_server.exe

Mirrors doudizhu_ai/server/tests/api_test.py: self-hosts the binary on a random
port unless --base-url is given, then walks the whole request/response contract.
"""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASSED = 0
FAILED = 0


def check(name, ok, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f"  <- {detail}" if detail else ""))


def request(base, method, path, body=None, sid=None, raw=None, timeout=10):
    headers = {}
    if sid:
        headers["X-Session-Id"] = sid
    data = None
    if raw is not None:
        data = raw.encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def get_json(base, method, path, **kw):
    status, headers, raw = request(base, method, path, **kw)
    try:
        return status, headers, json.loads(raw.decode("utf-8"))
    except Exception:
        return status, headers, None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def locate_exe(explicit):
    if explicit:
        p = pathlib.Path(explicit)
        return p if p.is_absolute() else ROOT / p
    for name in ("build_gcc", "build_msvc"):
        for exe in ("tetris_ai_server.exe", "tetris_ai_server"):
            p = ROOT / name / exe
            if p.exists():
                return p
    return None


def start_server(exe, port):
    env = os.environ.copy()
    # MSVC build links the CUDA LibTorch DLLs from the active Python install.
    try:
        import torch
        env["PATH"] = str(pathlib.Path(torch.__file__).parent / "lib") + os.pathsep + env["PATH"]
    except Exception:
        pass
    proc = subprocess.Popen(
        [str(exe), "--port", str(port), "--seed", "12345"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def wait_ready(base, proc, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            status, _, _ = request(base, "GET", "/api/status", timeout=1)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def run(base):
    print(f"\n== contract @ {base} ==\n")

    status, headers, dto = get_json(base, "GET", "/api/status")
    check("status 200", status == 200, str(status))
    check("status DTO", isinstance(dto, dict) and dto.get("status") == "ok", repr(dto)[:120])
    check("status exposes model capability", isinstance(dto, dict) and "modelAvailable" in dto)
    check("CORS header", headers.get("Access-Control-Allow-Origin") == "*", str(headers))

    status, headers, raw = request(base, "GET", "/")
    check("index.html served", status == 200 and b"TETRIS AI" in raw, str(status))
    check("index MIME", "text/html" in headers.get("Content-Type", ""), headers.get("Content-Type", ""))
    status, js_headers, raw = request(base, "GET", "/app.js")
    check("app.js served", status == 200 and b"api/game/step" in raw, str(status))
    check("app.js MIME", "javascript" in js_headers.get("Content-Type", ""),
          js_headers.get("Content-Type", ""))
    status, css_headers, raw = request(base, "GET", "/style.css")
    check("style.css served", status == 200 and b"--accent" in raw, str(status))
    check("style.css MIME", "text/css" in css_headers.get("Content-Type", ""),
          css_headers.get("Content-Type", ""))

    status, _, _ = request(base, "OPTIONS", "/api/game/step")
    check("OPTIONS 204", status == 204, str(status))

    sid = "test_session_a"
    status, headers, dto = get_json(base, "POST", "/api/game/new",
                                    body={"seed": 4242, "mode": "model"}, sid=sid)
    check("new 200", status == 200 and isinstance(dto, dict) and not dto.get("error"), repr(dto)[:140])
    check("session echoed", headers.get("X-Session-Id") == sid, headers.get("X-Session-Id", ""))
    check("board shape 20x10",
          isinstance(dto.get("board"), list) and len(dto["board"]) == 20
          and all(len(r) == 10 for r in dto["board"]))
    check("board cell range",
          all(0 <= v <= 7 for r in dto["board"] for v in r))
    check("initial DTO", dto.get("pieces") == 0 and dto.get("step") == 0
          and dto.get("active") is not None and len(dto.get("next", [])) == 5)
    check("ghost present", dto.get("ghost") is not None)
    check("metrics present", isinstance(dto.get("metrics"), dict)
          and len(dto["metrics"].get("heights", [])) == 10)

    _, _, other = get_json(base, "POST", "/api/game/new",
                           body={"seed": 999, "mode": "heuristic"}, sid="test_session_b")
    _, _, again = get_json(base, "GET", "/api/game/state", sid=sid)
    check("sessions isolated", again.get("seed") == 4242 and other.get("seed") == 999,
          f"{again.get('seed')} / {other.get('seed')}")

    status, _, dto = get_json(base, "POST", "/api/game/step", body={"count": 1}, sid=sid)
    check("step 200", status == 200 and not dto.get("error"), repr(dto)[:140])
    check("step advances pieces", dto.get("pieces") == 1 and dto.get("step") == 1,
          f"pieces={dto.get('pieces')} step={dto.get('step')}")
    check("decision present", isinstance(dto.get("decision"), dict)
          and dto["decision"].get("source") in ("model", "heuristic", "harddrop"))
    a = dto.get("animation") or {}
    d = dto.get("decision") or {}
    check("animation frames", isinstance(a.get("frames"), list) and len(a["frames"]) >= 2,
          str(len(a.get("frames", []))))
    check("animation boardBefore 20x10", len(a.get("boardBefore", [])) == 20)
    check("animation clears match decision",
          a.get("linesCleared") == d.get("linesCleared"),
          f"{a.get('linesCleared')} / {d.get('linesCleared')}")
    frames = a.get("frames", [])
    check("frames end at lock", frames and frames[-1]["y"] >= frames[0]["y"],
          f"{frames[0]['y'] if frames else '?'} -> {frames[-1]['y'] if frames else '?'}")
    check("lockedCells have 4 cells", len(a.get("lockedCells", [])) == 4)

    _, _, before = get_json(base, "GET", "/api/game/state", sid=sid)
    _, _, burst = get_json(base, "POST", "/api/game/step", body={"count": 10}, sid=sid)
    check("burst advances 10", burst.get("pieces") == before.get("pieces") + 10,
          f"{before.get('pieces')} -> {burst.get('pieces')}")
    check("burst keeps valid board", len(burst.get("board", [])) == 20)

    _, _, ticked = get_json(base, "POST", "/api/game/tick",
                            body={"ops": ["Right", "RotateCW"]}, sid=sid)
    check("tick 200", not ticked.get("error") and isinstance(ticked.get("board"), list),
          repr(ticked)[:120])
    check("tick advances step", ticked.get("step") == burst.get("step") + 1,
          f"{burst.get('step')} -> {ticked.get('step')}")
    check("tick drops animation", "animation" not in ticked)

    _, _, bad_json = get_json(base, "POST", "/api/game/step", raw="{not json", sid=sid)
    check("invalid body does not crash", isinstance(bad_json, dict) and "board" in bad_json,
          repr(bad_json)[:120])

    status, _, _ = request(base, "GET", "/%2e%2e/CMakeLists.txt")
    check("path traversal rejected", status == 404, str(status))
    status, _, body = request(base, "GET", "/api/nope")
    check("unknown endpoint 404", status == 404, str(status))
    status, _, _ = request(base, "POST", "/api/game/state", body={}, sid=sid)
    check("wrong method 404/405", status in (404, 405), str(status))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="")
    ap.add_argument("--exe", default="")
    args = ap.parse_args()

    proc = None
    if args.base_url:
        base = args.base_url.rstrip("/")
    else:
        exe = locate_exe(args.exe)
        if not exe or not exe.exists():
            print(f"server binary not found: {exe}")
            return 1
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        print(f"starting {exe} on port {port}")
        proc = start_server(exe, port)
        if not wait_ready(base, proc):
            print("server did not become ready")
            if proc:
                proc.terminate()
            return 1

    try:
        run(base)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    print(f"\n{PASSED} passed, {FAILED} failed\n")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
