"""Loopback-only byte-range video transport. Only registered files are served."""

import hashlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class VideoServer:
    def __init__(self):
        self.files = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = owner.files.get(self.path)
                if path is None:
                    self.send_error(404)
                    return
                size = path.stat().st_size
                start, end = 0, size - 1
                header = self.headers.get("Range")
                if header:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
                    if not match or not any(match.groups()):
                        self.send_error(416)
                        return
                    a, b = match.groups()
                    if a:
                        start = int(a)
                        end = min(int(b), size - 1) if b else size - 1
                    else:
                        start = max(0, size - int(b))
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                self.send_response(206 if header else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(end - start + 1))
                if header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with path.open("rb") as file:
                        file.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            data = file.read(min(1024 * 1024, remaining))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def register(self, path):
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise ValueError("Supply a local MP4 file")
        token = "/" + hashlib.sha256(str(path).encode()).hexdigest() + ".mp4"
        self.files[token] = path
        return f"http://127.0.0.1:{self.server.server_port}{token}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
