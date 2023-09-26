from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import warnings

import torch
import numpy as np
import tqdm

from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, CHUNK_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, WriteTXT, WriteVTT, WriteSRT
from whisper import load_model
from whisper.model import Whisper

class WhisperStreaming():

    def __init__(self):

        # 引数から移植
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model("medium", device=self.device)

        self.verbose = True
        self.temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.compression_ratio_threshold: Optional[float] = 2.4
        self.logprob_threshold: Optional[float] = -1.0
        self.no_speech_threshold: Optional[float] = 0.6
        self.condition_on_previous_text: bool = True

        # 今回はlanguageを日本語に固定、taskもtranscribeに固定
        self.decode_options = {
            "language": "ja",
            "task": "transcribe",
            "fp16": True,
        }

        # transcribeのループ外のものを移植
        self.dtype = torch.float16 if self.decode_options.get("fp16", True) else torch.float32
        if self.model.device == torch.device("cpu"):
            if torch.cuda.is_available():
                warnings.warn("Performing inference on CPU when CUDA is available")
            if self.dtype == torch.float16:
                warnings.warn("FP16 is not supported on CPU; using FP32 instead")
                self.dtype = torch.float32

        if self.dtype == torch.float32:
            self.decode_options["fp16"] = False

        language = self.decode_options["language"]
        task = self.decode_options.get("task", "transcribe")
        self.tokenizer = get_tokenizer(self.model.is_multilingual, language=language, task=task)

        self.seek = 0
        self.input_stride = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )  # mel frames per output token: 2
        self.time_precision = (
            self.input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        self.all_tokens = []
        self.all_segments = []
        self.prompt_reset_since = 0

        # バッファ記憶用のメンバ変数
        self.buffer = b''

    def set_data(self, audio: bytes):

        # 前回の残り分と連結する
        self.buffer = self.buffer + audio

        # 30秒以下になるまでループする
        while len(self.buffer) >= SAMPLE_RATE*CHUNK_LENGTH*2:

            # 30秒分バッファから取得
            frame = self.buffer[:SAMPLE_RATE*CHUNK_LENGTH*2]

            # byteデータを2byteデータに詰めなおして最大値で割って-1～1に正規化する
            frame = np.frombuffer(frame, np.int16).flatten().astype(np.float32) / 32768.0

            # 処理前の場所を記憶しておく
            previous_seek = self.seek

            # 30秒データを処理する
            self.set_frame(frame)

            # 処理済みとなった時刻がself.seekに記載されるため
            # 処理済みをバッファから消す更新を行う
            self.buffer = self.buffer[(self.seek - previous_seek)*HOP_LENGTH*2:]

        return

    def set_frame(self, audio: bytes):

        # 音響特徴量計算
        mel = log_mel_spectrogram(audio)
        mel = mel.unsqueeze(0)

        # 現時刻のオフセットをseekから計算
        timestamp_offset = float(self.seek * HOP_LENGTH / SAMPLE_RATE)
        segment = pad_or_trim(mel, N_FRAMES).to(self.device).to(self.dtype)
        segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

        self.decode_options["prompt"] = self.all_tokens[self.prompt_reset_since:]
        result = self.decode_with_fallback(segment)[0]
        tokens = torch.tensor(result.tokens)

        # 有音無音判定
        if self.no_speech_threshold is not None:
            # no voice activity check
            should_skip = result.no_speech_prob > self.no_speech_threshold
            if self.logprob_threshold is not None and result.avg_logprob > self.logprob_threshold:
                # don't skip if the logprob is high enough, despite the no_speech_prob
                should_skip = False

            if should_skip:
                self.seek += segment.shape[-1]  # fast-forward to the next segment boundary
                return

        # トークナイザでデコード
        timestamp_tokens: torch.Tensor = tokens.ge(self.tokenizer.timestamp_begin)

        # タイムスタンプトークンによる区間処理
        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
        if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = (
                    sliced_tokens[0].item() - self.tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                    sliced_tokens[-1].item() - self.tokenizer.timestamp_begin
                )
                self.add_segment(
                    start=timestamp_offset + start_timestamp_position * self.time_precision,
                    end=timestamp_offset + end_timestamp_position * self.time_precision,
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                )
                last_slice = current_slice
            last_timestamp_position = (
                tokens[last_slice - 1].item() - self.tokenizer.timestamp_begin
            )
            self.seek += last_timestamp_position * self.input_stride
            self.all_tokens.extend(tokens[: last_slice + 1].tolist())

        # 区間が無い場合の処理
        else:
            duration = segment_duration
            timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            if len(timestamps) > 0:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = timestamps[-1].item() - self.tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            self.add_segment(
                start=timestamp_offset,
                end=timestamp_offset + duration,
                text_tokens=tokens,
                result=result,
            )

            self.seek += segment.shape[-1]
            self.all_tokens.extend(tokens.tolist())

        if not self.condition_on_previous_text or result.temperature > 0.5:
            # do not feed the prompt tokens if a high temperature was used
            self.prompt_reset_since = len(self.all_tokens)

        return

    def decode_with_fallback(self, segment: torch.Tensor) -> List[DecodingResult]:
        temperatures = [self.temperature] if isinstance(self.temperature, (int, float)) else self.temperature
        kwargs = {**self.decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results = self.model.decode(segment, options)

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                self.compression_ratio_threshold is not None
                and result.compression_ratio > self.compression_ratio_threshold
                or self.logprob_threshold is not None
                and result.avg_logprob < self.logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries = self.model.decode(segment[needs_fallback], options)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]

        return results

    def add_segment(self,
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = self.tokenizer.decode([token for token in text_tokens if token < self.tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        self.all_segments.append(
            {
                "id": len(self.all_segments),
                "seek": self.seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if self.verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")
        
        return

    def finalized(self):
        frame = self.buffer

        # byteデータを2byteデータに詰めなおして最大値で割って-1～1に正規化する
        frame = np.frombuffer(frame, np.int16).flatten().astype(np.float32) / 32768.0

        # 内部で0詰めして30秒データとしてから処理
        self.set_frame(frame)

    def get_result(self):
        return dict(
            text=self.tokenizer.decode(self.all_tokens), 
            segments=self.all_segments
        )