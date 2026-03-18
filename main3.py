import sys, math, threading
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QTabletEvent, QPainter, QColor, QPen

# ── 설정 ─────────────────────────────
RATE = 44100
CHUNK = 128
BASE_FREQ = 130.81  # C3
OCTAVES = 3
MAX_VOICES = 5

NOTE_PATTERN = ['C','D','E','F','G','A','B']
NOTE_DIST = [1,1,0.5,1,1,1,0.5]
TOTAL_POS = sum(NOTE_DIST)*OCTAVES

NOTE_COLORS = [
    QColor(255,127,0, 80),    # D 주
    QColor(255,255,0, 80),    # E 노
    QColor(0,255,0, 80),      # F 초
    QColor(0,0,255, 80),      # G 파
    QColor(75,0,130, 80),     # A 보
    QColor(148,0,211, 80),    # B 핑
    QColor(255,0,0, 80),      # C 빨
]

# ── 보이스 구조
_voices = {}  # {finger_id: {freq, amp, phase, target_amp}}
_lock = threading.Lock()

FADE_TIME = 0.05  # 페이드아웃

# ── 좌표 → 주파수
def pos_to_freq(nx, ny):
    t = np.clip((nx + ny)/2,0,1)
    freq = BASE_FREQ * 2**(t * OCTAVES)
    return freq

# ── 오디오 콜백
def audio_callback(outdata, frames, time_info, status):
    buf = np.zeros(frames, dtype=np.float32)
    remove_keys = []
    with _lock:
        for fid, v in _voices.items():
            phase = v['phase']
            amp = v['amp']
            target_amp = v.get('target_amp', amp)
            freq = v['freq']
            for i in range(frames):
                # 페이드 적용
                if amp > target_amp:
                    amp -= (amp / (FADE_TIME*RATE))
                    amp = max(amp, 0)
                else:
                    amp = target_amp
                buf[i] += amp * math.sin(phase)
                phase += 2*math.pi*freq / RATE
                if phase > 2*math.pi:
                    phase -= 2*math.pi
            v['phase'] = phase
            v['amp'] = amp
            if amp < 0.001 and target_amp==0:
                remove_keys.append(fid)
        for k in remove_keys:
            del _voices[k]
    n = max(1,len(_voices))
    outdata[:,0] = np.clip(buf/n, -1.0,1.0)

# ── GUI / 터치
class TabletSynth(QWidget):
    def tabletEvent(self, event: QTabletEvent):
        fid = 0
        nx = event.x()/self.width()
        ny = 1 - event.y()/self.height()
        freq = pos_to_freq(nx, ny)
        with _lock:
            phase = _voices.get(fid,{}).get('phase',0)
            _voices[fid] = {'freq':freq,'amp':0.4,'phase':phase,'target_amp':0.4}
        self.update()

    def mouseReleaseEvent(self, event):
        with _lock:
            for v in _voices.values():
                v['target_amp'] = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w,h = self.width(), self.height()
        diag_start = np.array([0,h])
        diag_end   = np.array([w,0])

        # 음 위치 누적
        positions = []
        for o in range(OCTAVES):
            pos_accum = 0
            for d in NOTE_DIST:
                pos_accum += d
                positions.append(pos_accum + o*sum(NOTE_DIST))

        for i, pos in enumerate(positions):
            note_idx = i%7
            color = NOTE_COLORS[note_idx]
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)

            t = pos/TOTAL_POS
            px = diag_start[0]*(1-t)+diag_end[0]*t
            py = diag_start[1]*(1-t)+diag_end[1]*t

            # 45° 선
            angle = math.radians(45)
            nx_ = math.cos(angle)
            ny_ = math.sin(angle)
            line_len = max(w,h)*1.5
            x1 = px - nx_*line_len
            y1 = py - ny_*line_len
            x2 = px + nx_*line_len
            y2 = py + ny_*line_len
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

# ── 메인
stream = sd.OutputStream(samplerate=RATE, blocksize=CHUNK,
                         channels=1,dtype='float32',
                         callback=audio_callback)
stream.start()

app = QApplication(sys.argv)
w = TabletSynth()
w.resize(500,500)
w.setWindowTitle("RTS-300 스타일로폰 (C~B 색상, 5ms 페이드아웃)")
w.show()
sys.exit(app.exec_())