#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import signal as syssig
import sys

def handler(signal, frame):
    raise SystemExit('Exiting')


import math

import argparse

import numpy as np
import pandas as pd
import wave
import pyworld as pw
from scipy import signal
from scipy import interpolate

# by https://gist.github.com/jgraving/e9e0e490ed83f84501d38061f1f985f2
def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.nanmedian(y, axis=1)

def maxfilt(x, k):
    """Apply a length-k maximum filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "maximum filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.max(y, axis=1)

# by https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
def pandas_fill(arr, method="ffill"):
    df = pd.Series(arr)
    # df.fillna(method=method, inplace=True)
    if method == "ffill":
        df.ffill(inplace=True)
    if method == "bfill":
        df.bfill(inplace=True)

    return df.values

def pandas_interpolate(arr):
    df = pd.Series(arr)
    df = df.interpolate()
    return df.values

def analysis_resynthesis(input_signal, sampling_rate, min_freq, max_freq, frame_period, speed, use_harvest, d4c_threshold, world_fft_size, f0_rate, fix_f0, sp_rate, sp_pow, aperiodicity_offset):

    # 音響特徴量の抽出
    if not use_harvest:
        print("dio...")
        f0, t = pw.dio(x=input_signal, fs=sampling_rate, f0_floor=min_freq,
                       f0_ceil=max_freq, frame_period=frame_period)  # 基本周波数の抽出
    else:
        print("harvest...")
        f0, t = pw.harvest(x=input_signal, fs=sampling_rate, f0_floor=min_freq,
                           f0_ceil=max_freq, frame_period=frame_period)  # 基本周波数の抽出

    print("stonemask...")
    f0 = pw.stonemask(input_signal, f0, t, sampling_rate)  # refinement

    # f0の推定に失敗すると0が返ってくるのでそれを補間
    print("fix...")
    f0[f0 == 0] = 'nan'  # f0の0をnanに置き換え
    # f0 = pandas_interpolate(f0) # f0の配列のnanを前後から線形補間 あまり良くない
    f0 = pandas_fill(f0, "ffill")  # f0の配列のnanを前の要素で補間
    f0 = pandas_fill(f0, "bfill")  # f0の配列のnanを後の要素で補間 念の為
    f0 = np.nan_to_num(f0)  # f0のnanを0に置き換え f0にnanがあると壊れる
    f0 = medfilt(f0, 5)  # メディアンフィルタでノイズ除去

    print("cheaptrick...")
    sp = pw.cheaptrick(input_signal, f0, t, sampling_rate, fft_size=world_fft_size)  # スペクトル包絡の抽出

    print("d4c...")
    # thresholdはデフォルトで0.85 0 ~ 1 下げるほど有声音として判断される
    # ノイズが多い入力では0.65あたりまで下げたほうが雑音が入らない
    ap = pw.d4c(input_signal, f0, t, sampling_rate, threshold=d4c_threshold, fft_size=world_fft_size)  # 非周期性指標の抽出

    # 全て無声音とする(ささやき化)
    # ap = ap + 1

    ap += aperiodicity_offset

    # フィルターを無効化
    # sp = 0 * sp + 131072

    # ピッチの変更
    print("pitch shift...")

    modified_f0 = np.copy(f0)

    # 固定ピッチ
    if fix_f0 != 0 and (not math.isnan(fix_f0)):
        if True:
            # 固定ピッチに強制
            modified_f0 = (modified_f0 * 0) + fix_f0
        else:
            # 平均を除去した上で固定ピッチを加算
            freq = 5
            filt = signal.butter(N=1, Wn=freq, btype="lowpass", output='sos', fs=(1000 / frame_period))
            mean = signal.sosfilt(sos=filt, x=modified_f0)
            modified_f0 = (modified_f0 - mean) + fix_f0
    
    # ピッチチェンジ
    modified_f0 = modified_f0 * f0_rate

    # フォルマントを拡大縮小
    print("formant shift...")

    modified_sp = np.zeros_like(sp)
    if sp_rate <= 1:
        # 縮小 足りない高周波は前の要素で補間
        change_x = np.arange(modified_sp.shape[1])

        # フォルマントをpowカーブで変形 sp_powは低いと声が低くなる
        change_x = np.power(change_x / modified_sp.shape[1], sp_pow) * modified_sp.shape[1]

        # フォルマント全体変形
        change_x /= sp_rate

        for i in range(modified_sp.shape[0]):
            f = interpolate.interp1d(x=np.arange(
                modified_sp.shape[1]), y=sp[i, :], kind='linear', axis=0, copy=False, bounds_error=False)
            modified_sp[i, :] = f(change_x)
            modified_sp[i, :] = pandas_fill(modified_sp[i, :], "ffill")  # NaNを前の要素で補間
        
        # NaNの値を0にする
        modified_sp = np.nan_to_num(modified_sp)
    else:
        # 拡大 高周波は切り捨て
        change_x = np.arange(modified_sp.shape[1])

        # フォルマントをpowカーブで変形 sp_powは低いと声が低くなる
        change_x = np.power(change_x / modified_sp.shape[1], sp_pow) * modified_sp.shape[1]

        # フォルマント全体変形
        change_x /= sp_rate

        for i in range(modified_sp.shape[0]):
            f = interpolate.interp1d(x=np.arange(
                modified_sp.shape[1]), y=sp[i, :], kind='linear', axis=0, copy=False, bounds_error=False)
            modified_sp[i, :] = f(change_x)
        
        # NaNの値を0にする
        modified_sp = np.nan_to_num(modified_sp)
        pass

    # 再合成
    print("synthesize...")
    synth = pw.synthesize(modified_f0, modified_sp, ap, sampling_rate, frame_period=frame_period * speed)

    return synth


if __name__ == "__main__":
    syssig.signal(syssig.SIGINT, handler)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input WAV file path. Audio file must be mono. It is required.", required=True)
    parser.add_argument("-o", "--output", help="Output WAV file path. It is required.", required=True)
    parser.add_argument(
        "-p", "--f0", help="F0 Frequency multiplier. No change in 1.0. Value must be greater than 0. Increasing this value increases the pitch. default is %(default)s", type=float, default=1.0)
    parser.add_argument(
        "-f", "--formant", help="Formant Frequency multiplier. No change in 1.0. Value must be greater than 0. Increasing this value increasing the pitch of the formant. default is %(default)s", type=float, default=1.0)
    parser.add_argument(
        "-c", "--formant_pow", help="inverse of Formant frequency power curve. No change in 1.0. Value must be greater than 0. Increasing this value increasing the pitch of the formant. default is %(default)s", type=float, default=1.0)
    parser.add_argument(
        "-F", "--fix_f0", help="Fixed F0 frequency. Enabled if other than 0. default is %(default)s", type=float, default=0.0)
    parser.add_argument("-s", "--speed", help="Speed multiplier to change. No change in 1.0. Value must be greater than 0. Increasing the value will make it slower, decreasing it will make it faster. default is %(default)s", default=1.0, type=float)
    parser.add_argument("-v", "--main_volume",
                        help="Input volume multiplier. No change in 1.0. 0.5 to half, 2 to double. default is %(default)s", type=float, default=1.0)
    parser.add_argument(
        "-l", "--min_f0", help="F0 lowest frequency. Value unit is Hz. For example, a male voice is 80Hz, a female voice is 250Hz. default is %(default)s", type=float, default=80.0)
    parser.add_argument(
        "-H", "--max_f0", help="F0 highest frequency. Value unit is Hz. For example, a male voice is 800Hz, a female voice is 1500Hz. default is %(default)s", type=float, default=1500.0)
    parser.add_argument(
        "-a", "--aperiodicity", help="Aperiodicity offset. No change in 0.0. The value above 1 will make the sound unvoiced, and setting the value below -1 will make it voiced. default is %(default)s", type=float, default=0.0)
    parser.add_argument(
        "-t", "--d4c_threshold", help="Threshold for D4C aperiodicity-based voiced/unvoiced decision. If the value is 0, all sounds will be considered voiced, and if the value is 1, all sounds will be considered unvoiced. default is %(default)s", type=float, default=0.85)
    parser.add_argument("-w", "--fft_size", help="FFT window size. The value must be a power of 2. 2048 for high speed. Higher values improve quality, but slow processing. default is %(default)s", type=int, default=16384)
    parser.add_argument("-P", "--frame_period", help="Processing frame period time. The value is in milliseconds. lower the value, the more accurate the time becomes, but processing will be much slower. default is %(default)s", default=5.0, type=float)
    parser.add_argument("-d", "--use_dio", help="Use dio F0 extraction algorithm instead of Harvest. Slightly faster speed, but less accurate with noisy audio. default is %(default)s", action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    # parser.print_help()

    print("loading...")

    input_audio = wave.open(args.input, mode='rb')
    sampling_rate = input_audio.getframerate()
    audio = np.frombuffer(input_audio.readframes(-1), dtype="int16").astype(np.float64)
    input_audio.close()

    output = analysis_resynthesis(input_signal=audio * args.main_volume, sampling_rate=sampling_rate, min_freq=args.min_f0, max_freq=args.max_f0,
                                  frame_period=args.frame_period, speed=args.speed, d4c_threshold=args.d4c_threshold, use_harvest=(not args.use_dio), world_fft_size=args.fft_size, f0_rate=args.f0, fix_f0=args.fix_f0, sp_rate=args.formant, sp_pow=args.formant_pow, aperiodicity_offset=args.aperiodicity)

    output = output.clip(-32767, 32767)

    print("saving...")

    output_wave = wave.open(args.output, mode='wb')
    output_wave.setnchannels(1)  # モノラル
    output_wave.setsampwidth(2)  # 16bit/sample
    output_wave.setframerate(sampling_rate)
    output_wave.writeframes(output.astype(np.int16).tobytes())
    output_wave.close()

    print("done!")
