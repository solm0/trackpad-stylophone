#!/usr/bin/env python3
"""
트랙패드 스타일로폰 (폴리포닉)
- 여러 손가락 동시 접촉 → 화음
- 손가락마다 독립적인 오실레이터
- 왼쪽 하단(C3) ↔ 오른쪽 상단(C5), 2옥타브 sin파

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
CHUNK     = 1
BASE_FREQ = 130.81
SEMITONES = 24
NOTES     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAX_VOICES = 5  # 최대 동시 손가락 수

# ── 보이스 구조 ───────────────────────────────────────────────
# voices = { finger_id: { freq, target_freq, phase, amp } }
_lock   = threading.Lock()
_voices = {}   # 현재 활성 보이스

# ── 오디오 콜백 ───────────────────────────────────────────────
def audio_callback(outdata, frames, time_info, status):
    with _lock:
        voices_snapshot = {fid: dict(v) for fid, v in _voices.items()}

    buf = np.zeros(frames, dtype=np.float32)

    with _lock:
        for fid, v in _voices.items():
            for i in range(frames):
                # 글라이드
                v['freq'] += (v['target_freq'] - v['freq']) * 0.015
                # 어택
                v['amp']  += (v['target_amp'] - v['amp']) * (0.3 if v['target_amp'] > 0 else 0.04)
                v['phase'] += 2.0 * math.pi * v['freq'] / RATE
                if v['phase'] > 2.0 * math.pi:
                    v['phase'] -= 2.0 * math.pi
                buf[i] += v['amp'] * math.sin(v['phase'])

        # 릴리즈 중인 보이스 (target_amp=0, amp≈0) 제거
        dead = [fid for fid, v in _voices.items()
                if v['target_amp'] == 0 and abs(v['amp']) < 0.001]
        for fid in dead:
            del _voices[fid]

    # 보이스 수에 따라 볼륨 정규화
    n = max(1, 1)
    buf *= (0.4 / n)

    outdata[:, 0] = np.clip(buf, -1.0, 1.0)

# ── 좌표 → 주파수 ─────────────────────────────────────────────
def pos_to_freq(nx, ny):
    t = max(0.0, min(1.0, (nx + ny) / 2.0))
    s = t * SEMITONES
    f = BASE_FREQ * 2 ** (s / 12.0)
    m = 48 + round(s)
    return f, NOTES[m % 12] + str(m // 12 - 1)

# ── MultitouchSupport ─────────────────────────────────────────
class MTPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class MTVector(ctypes.Structure):
    _fields_ = [("pos", MTPoint), ("vel", MTPoint)]

class MTFinger(ctypes.Structure):
    _fields_ = [
        ("frame",      ctypes.c_int),
        ("timestamp",  ctypes.c_double),
        ("identifier", ctypes.c_int),
        ("state",      ctypes.c_int),
        ("unknown1",   ctypes.c_int),
        ("unknown2",   ctypes.c_int),
        ("normalized", MTVector),
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
    ctypes.c_void_p,
    ctypes.POINTER(MTFinger),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
)

def touch_callback(device, fingers_ptr, n_fingers, timestamp, frame):
    active_ids = set()

    for i in range(n_fingers):
        f = fingers_ptr[i]
        if f.state != 4:
            continue

        fid = f.identifier
        active_ids.add(fid)
        nx  = f.normalized.pos.x
        ny  = f.normalized.pos.y
        freq, name = pos_to_freq(nx, ny)

        with _lock:
            if fid not in _voices:
                # 새 손가락 → 새 보이스 생성
                _voices[fid] = {
                    'freq':       freq,
                    'target_freq': freq,
                    'phase':      0.0,
                    'amp':        0.0,
                    'target_amp': 0.4,
                    'name':       name,
                }
            else:
                # 기존 손가락 → 주파수 업데이트
                _voices[fid]['target_freq'] = freq
                _voices[fid]['name']        = name

    # 더 이상 접촉 안 하는 손가락 → 릴리즈
    with _lock:
        for fid in list(_voices.keys()):
            if fid not in active_ids:
                _voices[fid]['target_amp'] = 0.0

    # 터미널 출력
    with _lock:
        names = [v['name'] for v in _voices.values() if v['target_amp'] > 0]
    if names:
        print(f"\r  {' + '.join(names):<30}", end='', flush=True)
    else:
        print(f"\r  {'—':<30}", end='', flush=True)

    return 0

# ── 메인 ─────────────────────────────────────────────────────
def main():
    MT = ctypes.CDLL(
        "/System/Library/PrivateFrameworks/"
        "MultitouchSupport.framework/MultitouchSupport"
    )
    CF = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )

    MT.MTDeviceCreateList.restype              = ctypes.c_void_p
    MT.MTDeviceCreateList.argtypes             = []
    MT.MTRegisterContactFrameCallback.restype  = None
    MT.MTRegisterContactFrameCallback.argtypes = [
        ctypes.c_void_p, CB_TYPE, ctypes.c_void_p
    ]
    MT.MTDeviceStart.restype  = None
    MT.MTDeviceStart.argtypes = [ctypes.c_void_p, ctypes.c_int]

    CF.CFArrayGetCount.restype         = ctypes.c_long
    CF.CFArrayGetCount.argtypes        = [ctypes.c_void_p]
    CF.CFArrayGetValueAtIndex.restype  = ctypes.c_void_p
    CF.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]

    stream = sd.OutputStream(
        samplerate=RATE, blocksize=CHUNK,
        channels=1, dtype='float32',
        callback=audio_callback,
    )
    stream.start()

    print("━" * 50)
    print("  트랙패드 스타일로폰 (폴리포닉)")
    print("  손가락을 올리면 소리  /  떼면 즉시 무음")
    print("  여러 손가락 = 화음")
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