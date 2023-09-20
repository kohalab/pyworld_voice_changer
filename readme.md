# pyworld_voice_changer

Pythonで書かれ.ogg音声ファイルに対してボイスチェンジャー処理をし.oggファイルに保存するソフトです。

本プログラムのベースにこちらのブログのプログラムコードを利用しました。感謝申し上げます。  
https://tam5917.hatenablog.com/entry/2019/04/28/123934

<details><summary>ベースになったプログラムの著作権表示</summary><div>

```
Copyright (c) 2020 Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
</div></details><br>

音声の処理には[WORLD 音声分析変換合成システム](http://www.isc.meiji.ac.jp/~mmorise/world/index.html)を使用して自然で高品質な変換ができます。

処理はF0周波数の乗算、フォルマント周波数の乗算、速度の変更、非周期性オフセットができます。

## Install
Python 3.10.6で動作を確認しています

依存関係のインストール
```
pip install -r requirements.txt
```

## Run
実行方法の例

ピッチが1.5倍、フォルマントが1/0.85倍
```
python . -i "in.ogg" -o "out.ogg" -p 1.5 -f 0.85
```

## Example
<!-- 
convert wav to ogg(aoTuV):
find . -name '*.wav' -print0 | xargs -0 -I {} oggenc2 -q 9 {} -o {}.ogg
-->
### Male Game Over
VG Voice - Announcer.ogg  
https://freesound.org/people/TheAtomicBrain/sounds/319141/  
CC0 License.

Original  
<audio src="media/ex_gameover_orig.ogg" controls></audio>

without option  
<audio src="media/ex_gameover.ogg" controls></audio>

`-p 0.5`  
<audio src="media/ex_gameover_p0.5.ogg" controls></audio>

`-p 0.75`  
<audio src="media/ex_gameover_p0.75.ogg" controls></audio>

`-p 2`  
<audio src="media/ex_gameover_p2.ogg" controls></audio>

`-p 3`  
<audio src="media/ex_gameover_p3.ogg" controls></audio>

`-f 0.5`  
<audio src="media/ex_gameover_f0.5.ogg" controls></audio>

`-f 0.75`  
<audio src="media/ex_gameover_f0.75.ogg" controls></audio>

`-f 1.5`  
<audio src="media/ex_gameover_f1.5.ogg" controls></audio>

`-f 2`  
<audio src="media/ex_gameover_f2.ogg" controls></audio>

`-s 0.5`  
<audio src="media/ex_gameover_s0.5.ogg" controls></audio>

`-s 2`  
<audio src="media/ex_gameover_s2.ogg" controls></audio>

`-p 2 -f 0.75`  
<audio src="media/ex_gameover_p2f0.75.ogg" controls></audio>

### Female ladies and gentlemen
Greetings, ladies and gentlemen!
https://freesound.org/people/MadamVicious/sounds/368378/  
CC0 License.

Original  
<audio src="media/ex_ladies_and_gentlemen_orig.ogg" controls></audio>

without option  
<audio src="media/ex_ladies_and_gentlemen.ogg" controls></audio>

`-p 0.5`  
<audio src="media/ex_ladies_and_gentlemen_p0.5.ogg" controls></audio>

`-p 0.75`  
<audio src="media/ex_ladies_and_gentlemen_p0.75.ogg" controls></audio>

`-p 2`  
<audio src="media/ex_ladies_and_gentlemen_p2.ogg" controls></audio>

`-f 0.5`  
<audio src="media/ex_ladies_and_gentlemen_f0.5.ogg" controls></audio>

`-f 0.75`  
<audio src="media/ex_ladies_and_gentlemen_f0.75.ogg" controls></audio>

`-f 1.5`  
<audio src="media/ex_ladies_and_gentlemen_f1.5.ogg" controls></audio>

`-f 2`  
<audio src="media/ex_ladies_and_gentlemen_f2.ogg" controls></audio>

`-s 0.5`  
<audio src="media/ex_ladies_and_gentlemen_s0.5.ogg" controls></audio>

`-s 2`  
<audio src="media/ex_ladies_and_gentlemen_s2.ogg" controls></audio>

`-p 0.5 -f 1.2`  
<audio src="media/ex_ladies_and_gentlemen_p0.5f1.2.ogg" controls></audio>

## Usage | help
```
usage: . [-h] -i INPUT -o OUTPUT [-p F0] [-f FORMANT] [-s SPEED] [-v MAIN_VOLUME] [-l MIN_F0] [-H MAX_F0]
         [-a APERIODICITY] [-t D4C_THRESHOLD] [-w FFT_SIZE] [-P FRAME_PERIOD] [-d]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input.ogg file path. Audio file must be mono. It is required.
  -o OUTPUT, --output OUTPUT
                        Output.ogg file path. It is required.
  -p F0, --f0 F0        F0 Frequency multiplier. No change in 1.0. Value must be greater than 0. Increasing this value
                        increases the pitch. default is 1.0
  -f FORMANT, --formant FORMANT
                        Formant Frequency multiplier. No change in 1.0. Value must be greater than 0. Increasing this
                        value decreases the pitch of the formant. default is 1.0
  -s SPEED, --speed SPEED
                        Speed multiplier to change. No change in 1.0. Value must be greater than 0. Increasing the value
                        will make it slower, decreasing it will make it faster. default is 1.0
  -v MAIN_VOLUME, --main_volume MAIN_VOLUME
                        Input volume multiplier. No change in 1.0. 0.5 to half, 2 to double. default is 1.0
  -l MIN_F0, --min_f0 MIN_F0
                        F0 lowest frequency. Value unit is Hz. For example, a male voice is 80Hz, a female voice is 250Hz.
                        default is 80.0
  -H MAX_F0, --max_f0 MAX_F0
                        F0 highest frequency. Value unit is Hz. For example, a male voice is 800Hz, a female voice is
                        1500Hz. default is 1500.0
  -a APERIODICITY, --aperiodicity APERIODICITY
                        Aperiodicity offset. No change in 0.0. The value above 1 will make the sound unvoiced, and setting
                        the value below -1 will make it voiced. default is 0.0
  -t D4C_THRESHOLD, --d4c_threshold D4C_THRESHOLD
                        Threshold for D4C aperiodicity-based voiced/unvoiced decision. If the value is 0, all sounds will
                        be considered voiced, and if the value is 1, all sounds will be considered unvoiced. default is
                        0.85
  -w FFT_SIZE, --fft_size FFT_SIZE
                        FFT window size. The value must be a power of 2. 2048 for high speed. Higher values improve
                        quality, but slow processing. default is 16384
  -P FRAME_PERIOD, --frame_period FRAME_PERIOD
                        Processing frame period time. The value is in milliseconds. lower the value, the more accurate the
                        time becomes, but processing will be much slower. default is 5.0
  -d, --use_dio         Use dio F0 extraction algorithm instead of Harvest. Slightly faster speed, but less accurate with
                        noisy audio. default is False
```

日本語版
```
usage: . [-h] -i INPUT -o OUTPUT [-p F0] [-f FORMANT] [-s SPEED] [-v MAIN_VOLUME] [-l MIN_F0] [-H MAX_F0]
         [-a APERIODICITY] [-t D4C_THRESHOLD] [-w FFT_SIZE] [-P FRAME_PERIOD] [-d]

options:
  -h, --help            このヘルプ メッセージを表示して終了します
  -i INPUT, --input INPUT
                        入力.oggファイルのパス。 音声ファイルはモノラルでなければなりません。 必須。
  -o OUTPUT, --output OUTPUT
                        出力.oggファイルのパス。 音声ファイルはモノラルでなければなりません。 必須。
  -p F0, --f0 F0        F0 周波数乗数。 1.0で変化なし。 値は 0 より大きくなければなりません。この値を増やすとピッチが高くなります。 デフォルトは1.0です
  -f FORMANT, --formant FORMANT
                        フォルマント周波数乗数。 1.0で変化なし。 値は 0 より大きくなければなりません。この値を大きくすると、フォルマントのピッチが低くなります。 デフォルトは1.0です
  -s SPEED, --speed SPEED
                        速度変更乗数。 1.0で変化なし。 値は 0 より大きくなければなりません。値を増やすと遅くなり、減らすと速くなります。 デフォルトは1.0です
  -v MAIN_VOLUME, --main_volume MAIN_VOLUME
                        入力ボリューム乗数。 1.0で変化なし。 0.5で半分、2で2倍。 デフォルトは1.0です
  -l MIN_F0, --min_f0 MIN_F0
                        F0 最低周波数。 値の単位は Hz です。 たとえば、男性の声は 80 Hz、女性の声は 250 Hz です。 デフォルトは80.0です
  -H MAX_F0, --max_f0 MAX_F0
                        F0 最高周波数。 値の単位は Hz です。 たとえば、男性の声は 800 Hz、女性の声は 1500 Hz です。 デフォルトは1500.0です
  -a APERIODICITY, --aperiodicity APERIODICITY
                        非周期性オフセット。 0.0で変化なし。 1 より大きい値を設定すると無声音になり、-1 より小さい値を設定すると有声音になります。 デフォルトは0.0です
  -t D4C_THRESHOLD, --d4c_threshold D4C_THRESHOLD
                        D4C の非周期性に基づく有声/無声の決定のしきい値。 値が 0 の場合、すべての音声は有声音とみなされ、値が 1 の場合、すべての音声は無声音とみなされます。 デフォルトは0.85です
  -w FFT_SIZE, --fft_size FFT_SIZE
                        FFTウィンドウのサイズ。 値は 2 のべき乗である必要があります。 高速にする場合は 2048。 値を大きくすると品質は向上しますが、処理が遅くなります。 デフォルトは16384です
  -P FRAME_PERIOD, --frame_period FRAME_PERIOD
                        処理フレーム周期時間。 値はミリ秒単位です。 値を小さくすると時間は正確になりますが、処理速度は非常に遅くなります。 デフォルトは5.0です
  -d, --use_dio         Harvest の代わりに DIO F0 抽出アルゴリズムを使用します。 速度はわずかに速くなりますが、ノイズの多い音声では精度が低くなります。 デフォルトは False です
```

## License
Distributed under the Unlicense License. See `LICENSE` for more information.