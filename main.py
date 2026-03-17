#!/usr/bin/env python3
"""
트랙패드 스타일로폰 (macOS 26 대응)
손가락 접촉 → 소리 ON, 뗌 → 즉시 소리 OFF
왼쪽 하단(C3) ↔ 오른쪽 상단(C5), 2옥타브 sin파

설치:
    pip install sounddevice numpy
실행:
    python3 trackpad_stylophone.py
"""

import sys, math, threading, time, ctypes

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print("pip install sounddevice numpy")
    sys.exit(1)

# ── 오디오 설정 ───────────────────────────────────────────────
RATE      = 44100
CHUNK     = 128
BASE_FREQ = 130.81   # C3
SEMITONES = 24       # 2옥타브
NOTES     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
SILENCE_TIMEOUT = 0.015

# ── 공유 상태 ─────────────────────────────────────────────────
_lock         = threading.Lock()
_touching     = False
_target_freq  = 0.0
_last_event_t = 0.0
_phase = 0.0
_freq  = 0.0
_amp   = 0.0

# ── 오디오 콜백 ───────────────────────────────────────────────
def audio_callback(outdata, frames, time_info, status):
    global _phase, _freq, _amp
    with _lock:
        on  = _touching
        tgt = _target_freq
    buf = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        _freq += (tgt - _freq) * 0.015
        _amp  += ((0.4 if on else 0.0) - _amp) * (0.01 if on else 0.04)
        _phase += 2.0 * math.pi * _freq / RATE
        if _phase > 2.0 * math.pi:
            _phase -= 2.0 * math.pi
        buf[i] = _amp * math.sin(_phase)
    outdata[:, 0] = buf

# ── 좌표 → 주파수 ─────────────────────────────────────────────
def pos_to_freq(nx, ny):
    t = max(0.0, min(1.0, (nx + ny) / 2.0))
    s = t * SEMITONES
    f = BASE_FREQ * 2 ** (s / 12.0)
    m = 48 + round(s)
    return f, NOTES[m % 12] + str(m // 12 - 1)

# ── MultitouchSupport 구조체 ──────────────────────────────────
class MTPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class MTVector(ctypes.Structure):
    _fields_ = [("pos", MTPoint), ("vel", MTPoint)]

class MTFinger(ctypes.Structure):
    _fields_ = [
        ("frame",      ctypes.c_int),
        ("timestamp",  ctypes.c_double),
        ("identifier", ctypes.c_int),
        ("state",      ctypes.c_int),   # 4 = 접촉 중, 7 = 뗌
        ("unknown1",   ctypes.c_int),
        ("unknown2",   ctypes.c_int),
        ("normalized", MTVector),       # 0.0~1.0 정규화 좌표
        ("size",       ctypes.c_float),
        ("unknown3",   ctypes.c_int),
        ("angle",      ctypes.c_float),
        ("majorAxis",  ctypes.c_float),
        ("minorAxis",  ctypes.c_float),
        ("unknown4",   MTVector),
        ("unknown5",   ctypes.c_int),
        ("unknown6",   ctypes.c_int),
        ("zDensity",   ctypes.c_float),
    ]

CB_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,            # device
    ctypes.POINTER(MTFinger),   # fingers
    ctypes.c_int,               # nFingers
    ctypes.c_double,            # timestamp
    ctypes.c_int,               # frame
)

# ── 터치 콜백 ─────────────────────────────────────────────────
def touch_callback(device, fingers_ptr, n_fingers, timestamp, frame):
    global _touching, _target_freq, _last_event_t

    active = [fingers_ptr[i] for i in range(n_fingers)
              if fingers_ptr[i].state == 4]

    with _lock:
        if active:
            f  = active[0]
            nx = f.normalized.pos.x
            ny = f.normalized.pos.y  # y축 반전
            freq, name = pos_to_freq(nx, ny)
            _target_freq  = freq
            _touching     = True
            _last_event_t = time.time()
            bar = '▓' * int(nx * 30)
            print(f"\r  {name:4s}  {freq:6.1f} Hz  [{bar:<30}]", end='', flush=True)
        else:
            _touching     = False
            _target_freq  = 0.0
            print(f"\r  —      ——.— Hz  {'':34}", end='', flush=True)
    return 0

# ── 메인 ─────────────────────────────────────────────────────
def main():
    MT = ctypes.CDLL(
        "/System/Library/PrivateFrameworks/"
        "MultitouchSupport.framework/MultitouchSupport"
    )

    MT.MTDeviceCreateList.restype          = ctypes.c_void_p
    MT.MTDeviceCreateList.argtypes         = []
    MT.MTRegisterContactFrameCallback.restype  = None
    MT.MTRegisterContactFrameCallback.argtypes = [
        ctypes.c_void_p, CB_TYPE, ctypes.c_void_p
    ]
    MT.MTDeviceStart.restype  = None
    MT.MTDeviceStart.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # CFArray 함수는 CoreFoundation에서
    CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    CF.CFArrayGetCount.restype     = ctypes.c_long
    CF.CFArrayGetCount.argtypes    = [ctypes.c_void_p]
    CF.CFArrayGetValueAtIndex.restype  = ctypes.c_void_p
    CF.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]

    stream = sd.OutputStream(
        samplerate=RATE, blocksize=CHUNK,
        channels=1, dtype='float32',
        callback=audio_callback,
    )
    stream.start()

    print("━" * 50)
    print("  트랙패드 스타일로폰")
    print("  손가락을 올리면 소리  /  떼면 즉시 무음")
    print("  왼쪽 하단 = C3   오른쪽 상단 = C5")
    print("  종료: Ctrl+C")
    print("━" * 50 + "\n")

    dev_list = MT.MTDeviceCreateList()
    n_dev    = CF.CFArrayGetCount(dev_list)

    if n_dev == 0:
        print("트랙패드를 찾을 수 없습니다.")
        stream.stop()
        sys.exit(1)

    cb      = CB_TYPE(touch_callback)
    _cb_ref = cb  # GC 방지

    for i in range(n_dev):
        dev = CF.CFArrayGetValueAtIndex(dev_list, i)
        MT.MTRegisterContactFrameCallback(dev, cb, None)
        MT.MTDeviceStart(dev, 0)

    print(f"  트랙패드 {n_dev}개 연결됨\n")
    print("  ※ 입력 모니터링 권한이 필요할 수 있습니다\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n  종료합니다.")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()