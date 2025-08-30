# COMPOSER.py
#
# qt_chat_min.py — pip-only: PySide6 + QtWebEngine + markdown-it-py + MathJax
# pip install PySide6 markdown-it-py
# sudo apt update
# sudo apt install -y libxcb-cursor0

import sys
import socket
import ast
import re
import json
import time
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QComboBox, QLineEdit, QSplitter
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt, QEvent, QObject, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QShortcut, QKeySequence  # >>> NEW, QTextCursor
from markdown_it import MarkdownIt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
from PySide6.QtGui import QFont, QShortcut, QKeySequence, QTextCursor


from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# ---- Echo service config ----
ECHO_HOST = "127.0.0.1"
ECHO_PORT = 6502
NULL = b"\x00"

def ensure_counter_file():
    cfile = Path(".counter")
    if not cfile.exists():
        cfile.write_text("0", encoding="utf-8")

def count() -> int:
    cfile = Path(".counter")
    if not cfile.exists():
        cfile.write_text("0", encoding="utf-8")
        value = 1
    else:
        try:
            old = int(cfile.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            old = 0
        value = old + 1
    cfile.write_text(str(value), encoding="utf-8")
    return value

def _recv_until_null(sock, chunk=4096):
    buf = bytearray()
    while True:
        part = sock.recv(chunk)
        if not part:
            raise ConnectionError("connection closed before NULL terminator")
        i = part.find(NULL)
        if i != -1:
            buf.extend(part[:i])
            return bytes(buf)
        buf.extend(part)

def echo_roundtrip(text: str, host=ECHO_HOST, port=ECHO_PORT, connect_timeout=3.0, recv_timeout=None) -> str:
    data = text.encode("utf-8") + NULL
    with socket.create_connection((host, port), timeout=connect_timeout) as s:
        s.settimeout(recv_timeout)
        s.sendall(data)
        reply_bytes = _recv_until_null(s)
        if reply_bytes and reply_bytes[-1] == 0:
            reply_bytes = reply_bytes[:-1]
    return reply_bytes.decode("utf-8", errors="replace")

MJ_URL  = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MJ_DIR  = (Path(__file__).parent / ".assets" / "mathjax").resolve()
MJ_FILE = MJ_DIR / "tex-svg.js"

def _ensure_mathjax_file():
    if MJ_FILE.is_file():
        return
    try:
        MJ_DIR.mkdir(parents=True, exist_ok=True)
        with urlopen(MJ_URL, timeout=20) as r:
            js = r.read()
        MJ_FILE.write_bytes(js)
    except (URLError, HTTPError, OSError) as e:
        sys.stderr.write(
            "\n[FATAL] MathJax not cached and could not be downloaded.\n"
            f"        Tried: {MJ_URL}\n"
            f"        Error: {e}\n"
            f"        Fix: connect once, or place the file manually at {MJ_FILE}\n"
        )
        sys.exit(2)

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
<style>
  *, *::before, *::after { box-sizing: border-box; }

  html, body {
    margin:0;
    overflow-x: hidden;
    background:#000000; /* true black */
    color:#e6e6e6;
    font-family: system-ui, sans-serif;
    font-size: 13px;
    line-height: 1.25;
  }

  .container {
    max-width: 1800px;
    width: min(100vw, 92vw);
    margin: 0 auto;
    padding: 10px;
  }

  .block {
    border: 1px solid #222;      /* keep box borders visible on black */
    border-radius: 10px;
    padding: 10px;
    margin: 16px 0;
    overflow-x: hidden;
    background: #111;            /* keep your existing box backgrounds */
  }

  .block .meta {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }

  .pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    line-height: 1.4;
    border: 1px solid #333;
    background: #1f1f1f;
    color: #ddd;
  }

  .user  .pill.voice { background:#0f2233; border-color:#14324c; color:#b8d7ff; }
  .assistant .pill.voice { background:#1a1633; border-color:#2a2355; color:#d3ccff; }
  .user      { background:#181800; }
  .assistant { background:#001818; }


  pre, code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.2;
    white-space: pre-wrap;
    word-break: break-word;
  }

  img, video, canvas, svg, table, iframe { max-width: 100%; height: auto; }

  h1 { font-size: 1.4em; }
  h2 { font-size: 1.25em; }
  h3 { font-size: 1.15em; }
  h4 { font-size: 1.05em; }
  h5, h6 { font-size: 1.0em; }

  mjx-container { font-size: 90%; max-width: 100%; overflow-x: auto; }

  a { color: #9ecbff; text-decoration: underline; pointer-events: none; }
</style>

  <script>
    document.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); }, true);

    async function appendBlock(role, html, meta){
      const wrap = document.createElement('div');
      wrap.className = 'block ' + role;

      let header = '';
      if (meta && (meta.voice || meta.turn || meta.turn === 0)) {
        const v = meta.voice ? `<span class="pill voice">${meta.voice}</span>` : '';
        const t = (meta.turn || meta.turn === 0) ? `<span class="pill turn">Turn ${meta.turn}</span>` : '';
        header = `<div class="meta">${v}${t}</div>`;
      }

      wrap.innerHTML = header + html;
      document.getElementById('chat').appendChild(wrap);
      if (window.MathJax?.typesetPromise) { await MathJax.typesetPromise([wrap]); }
      window.scrollTo(0, document.body.scrollHeight);
    }
    window._appendBlock = appendBlock;
  </script>
</head>
<body>
  <div class="container" id="chat"></div>
  <script>
    // appendBlock defined above
  </script>
</body>
</html>
"""

_HTML_LOCAL = None
def _html_local():
    global _HTML_LOCAL
    if _HTML_LOCAL is None:
        _ensure_mathjax_file()
        _HTML_LOCAL = HTML.replace(
            'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"',
            'src="tex-svg.js"'
        )
    return _HTML_LOCAL

md = MarkdownIt("commonmark", {"html": False})

def md_to_html(text: str) -> str:
    return md.render(text)

def _eval_py_string(expr: str) -> str:
    expr = (expr or "")
    if expr.strip() == "":
        return ""
    val = ast.literal_eval(expr)
    if not isinstance(val, str):
        raise TypeError("Not a string literal")
    return val

def _load_lines(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f.readlines() if line.rstrip("\n") != ""]
    except FileNotFoundError:
        return []
    except OSError:
        return []

TURN_RE = re.compile(r"^\(Turn\s+(\d+)\)")

def _harvest_high_turn_from_transcript(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    ENDSEP = "~~~(end)~~~"
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        m = TURN_RE.match(line)
        if not m:
            continue
        if i >= 2 and lines[i - 1].strip() == "" and lines[i - 2].strip() == ENDSEP:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None

def sync_counter_from_transcript():
    tpath = Path(".transcript.txt")
    n = _harvest_high_turn_from_transcript(tpath)
    if n is not None:
        Path(".counter").write_text(str(n), encoding="utf-8")
    else:
        ensure_counter_file()

class EchoWorker(QObject):
    finished = Signal(str)   # echoed text
    failed = Signal(str)     # error message

    def __init__(self, send_text: str, recv_timeout: Optional[float]):
        super().__init__()
        self.send_text = send_text
        self.recv_timeout = recv_timeout

    def run(self):
        try:
            out = echo_roundtrip(self.send_text, recv_timeout=self.recv_timeout)
            self.finished.emit(out)
        except Exception as e:
            self.failed.emit(str(e))

class Chat(QWidget):
    def __init__(self):
        super().__init__()
        app_font = QFont()
        app_font.setPointSize(8)
        QApplication.instance().setFont(app_font)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self.setWindowTitle("SSM Context Composer")

        # Fullscreen state + shortcuts  --------------------------------------
        self._is_fullscreen = False
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.exit_fullscreen)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=QApplication.quit)

        # Right-side: raw transcript viewer
        self.transcript_path = Path(".transcript.txt")
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        mono = QFont()
        mono.setFamily("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        mono.setPointSize(9)
        self.transcript_view.setFont(mono)
        self.transcript_view.setStyleSheet("""
            QTextEdit {
                background: #000;
                color: #ddd;
                border: 1px solid #333;
            }
            QScrollBar:vertical, QScrollBar:horizontal { background: #0f0f0f; border: none; margin: 0px; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #3a3a3a; min-height: 20px; min-width: 20px; border-radius: 4px; }
            QScrollBar::add-line, QScrollBar::sub-line { background: none; width: 0px; height: 0px; }
        """)
        self._last_sig = None
        self._poll = QTimer(self)
        self._poll.setInterval(500)
        self._poll.timeout.connect(self._refresh_transcript)
        self._poll.start()

        # Left-side: existing UI
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.view = QWebEngineView()
        self.view.setHtml(_html_local(), QUrl.fromLocalFile(str(MJ_DIR) + "/"))

        sync_counter_from_transcript()

        cwd = Path.cwd()
        s1_file = cwd / ".s1.txt"
        s2_file = cwd / ".s2.txt"
        s3_file = cwd / ".s3.txt"

        self.s1 = QComboBox(); self.s1.setEditable(False)
        self.s2 = QComboBox(); self.s2.setEditable(False)
        self.s3 = QComboBox(); self.s3.setEditable(False)

        self.s1.addItem(""); self.s2.addItem(""); self.s3.addItem("")
        for t in _load_lines(s1_file): self.s1.addItem(t)
        for t in _load_lines(s2_file): self.s2.addItem(t)
        for t in _load_lines(s3_file): self.s3.addItem(t)

        phosphor = "#00ff66"
        combo_css = f"""
            QComboBox {{
                background: #000;
                color: {phosphor};
                border: 1px solid #1f1f1f;
                padding: 4px 6px;
                font-family: "Monospace";
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            QComboBox QAbstractItemView {{
                background: #0b0b0b;
                color: {phosphor};
                selection-background-color: #101d10;
                selection-color: {phosphor};
                border: 1px solid #222;
                font-family: "Monospace";
            }}
        """
        self.s1.setStyleSheet(combo_css)
        self.s2.setStyleSheet(combo_css)
        self.s3.setStyleSheet(combo_css)

        srow = QHBoxLayout()
        srow.addWidget(self.s1); srow.addWidget(self.s2); srow.addWidget(self.s3)

        self.input = QTextEdit()
        self.input.setPlaceholderText("Type Markdown…  (Shift+Enter = Send)")
        self.input.setFixedHeight(120)
        self.input.setStyleSheet("""
            QTextEdit { background: #111; color: #e6e6e6; border: 1px solid #333;
                        selection-background-color: #444; selection-color: #fff; }
            QTextEdit:focus { border: 1px solid #555; }
            QScrollBar:vertical, QScrollBar:horizontal { background: #1a1a1a; border: none; margin: 0px; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #444; min-height: 20px; min-width: 20px; border-radius: 4px; }
            QScrollBar::add-line, QScrollBar::sub-line { background: none; width: 0px; height: 0px; }
        """)

        self.prepend = QLineEdit()
        self.prepend.setPlaceholderText("Optional header to prepend (e.g., !primed standard onestep)")
        self.prepend.setClearButtonEnabled(True)
        self.prepend.setStyleSheet(f"""
            QLineEdit {{
                background: #000;
                color: {phosphor};
                border: 1px solid #1f1f1f;
                selection-background-color: #133a20;
                selection-color: {phosphor};
                padding: 6px 8px;
                font-family: "Monospace";
            }}
            QLineEdit:focus {{ border: 1px solid #2f2f2f; }}
        """)

        self.send_btn = QPushButton("Send")
        clear_btn = QPushButton("Clear")

        btn_css = """
            QPushButton {
                background: #0d0d0d;
                color: #e6e6e6;
                border: 1px solid #222;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #161616; }
            QPushButton:pressed { background: #0b0b0b; }
            QPushButton:disabled { color: #666; border-color: #222; background: #0b0b0b; }
        """
        self.send_btn.setStyleSheet(btn_css)
        clear_btn.setStyleSheet(btn_css)

        btns = QHBoxLayout(); btns.addWidget(self.send_btn); btns.addWidget(clear_btn)

        # Assemble LEFT pane
        left_layout.addWidget(self.prepend)
        left_layout.addWidget(self.view, 1)
        left_layout.addLayout(srow)
        left_layout.addWidget(self.input)
        left_layout.addLayout(btns)

        # Splitter
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left)
        splitter.addWidget(self.transcript_view)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(splitter)
        self.showMaximized()
        self._set_splitter_50_50(splitter)

        # Hooks
        self.send_btn.clicked.connect(self.on_send)
        clear_btn.clicked.connect(lambda: self.view.setHtml(_html_local(), QUrl.fromLocalFile(str(MJ_DIR) + "/")))
        self.input.installEventFilter(self)

        self._busy = False
        self._pending = None
        self._thread = None
        self._worker = None

        self._refresh_transcript(force=True)

    def toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True

    def exit_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
            self._is_fullscreen = False

    def _set_splitter_50_50(self, splitter: QSplitter):
        def later():
            w = splitter.size().width()
            splitter.setSizes([w // 2, w - (w // 2)])
        QTimer.singleShot(0, later)

    def _refresh_transcript(self, force: bool = False):
        p = self.transcript_path
        if p.exists():
            try:
                st = p.stat()
                sig = (st.st_size, int(st.st_mtime))
            except OSError:
                sig = None
        else:
            sig = None

        if not force and sig == self._last_sig:
            return

        self._last_sig = sig
        try:
            text = p.read_text(encoding="utf-8", errors="ignore") if sig else ""
        except Exception:
            text = ""

        if self.transcript_view.toPlainText() != text:
            self.transcript_view.setPlainText(text)
            cursor = self.transcript_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.transcript_view.setTextCursor(cursor)

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
                if ev.modifiers() & Qt.ShiftModifier:
                    self.on_send(); return True
                else:
                    return False
        return super().eventFilter(obj, ev)

    def append(self, role: str, content_md: str, voice: Optional[str] = None, turn: Optional[int] = None):
        html = md_to_html(content_md)
        meta_py = {"voice": voice, "turn": turn}
        meta_js = json.dumps(meta_py)
        self.view.page().runJavaScript(f"window._appendBlock({role!r}, {html!r}, {meta_js});")

    def _set_busy(self, flag: bool):
        self._busy = flag
        self.send_btn.setEnabled(not flag)
        self.input.setEnabled(not flag)
        self.s1.setEnabled(not flag); self.s2.setEnabled(not flag); self.s3.setEnabled(not flag)
        self.prepend.setEnabled(not flag)

    def on_send(self):
        if self._busy:
            return

        text = self.input.toPlainText()
        display_text = text
        if text == "":
            return

        is_cmd = text.startswith("/")

        s1_label_raw: Optional[str] = None
        s3_label_raw: Optional[str] = None

        S1 = S2 = S3 = ""
        turn_id_user: Optional[int] = None
        turn_id_asst: Optional[int] = None

        try:
            t1 = (self.s1.currentText() or "")
            t2 = (self.s2.currentText() or "")
            t3 = (self.s3.currentText() or "")

            S1 = _eval_py_string(t1) if t1.strip() else ""
            S2 = _eval_py_string(t2) if t2.strip() else ""
            S3 = _eval_py_string(t3) if t3.strip() else ""

            s1_label_raw = S1 or None
            s3_label_raw = S3 or None

            if not is_cmd and S1:
                n = count(); turn_id_user = n
                S1 = f"(Turn {n}) {S1}"
            if not is_cmd and S3:
                n = count(); turn_id_asst = n
                S3 = f"(Turn {n}) {S3}"

        except Exception as e:
            self.append("assistant", f"**S parse error:** {e}")
            return

        self.append("user", display_text, voice=s1_label_raw, turn=turn_id_user if not is_cmd else None)

        if not is_cmd:
            if S1: text = S1 + text
            if S2: text = text + S2
            if S3: text = text + S3

        self.input.clear()
        print('IN:"' + text + '"')

        raw_header = self.prepend.text()
        header = raw_header if raw_header.strip() else None
        send_text = (header + "\n" + text) if header is not None else text

        self._set_busy(True)
        self._pending = {
            "send_text": send_text,
            "body_text": text,
            "s3_label_raw": s3_label_raw,
            "turn_id_asst": turn_id_asst if not is_cmd else None,
            "is_cmd": is_cmd,
        }

        self._thread = QThread(self)
        self._worker = EchoWorker(send_text=send_text, recv_timeout=600.0)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)

        def _cleanup():
            self._thread.quit(); self._thread.wait()
            self._worker.deleteLater(); self._thread.deleteLater()
            self._thread = None; self._worker = None

        self._worker.finished.connect(_cleanup)
        self._worker.failed.connect(_cleanup)
        self._thread.start()

    def _on_worker_finished(self, echoed: str):
        print('OUT:"' + echoed + '"')

        asst_label = self._pending["s3_label_raw"]
        asst_turn  = self._pending["turn_id_asst"]
        self.append("assistant", echoed, voice=asst_label, turn=asst_turn)

        if not self._pending.get("is_cmd", False):
            body_text = self._pending["body_text"]
            to_write = (body_text or "") + (echoed or "")
            with open('.transcript.txt', 'a', encoding='utf-8') as f:
                f.write(to_write)

        self._pending = None
        self._set_busy(False)
        self._refresh_transcript(force=True)

    def _on_worker_failed(self, msg: str):
        self.append("assistant", f"**ERROR:** {msg}")
        self._pending = None
        self._set_busy(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Chat(); w.showMaximized()
    sys.exit(app.exec())
