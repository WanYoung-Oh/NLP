"""
PRD 5-1 단위 테스트 — 모든 항목을 커버하는 자동화 테스트.

실행:
    cd /data/ephemeral/home/NLP
    python -m pytest tests/test_pipeline.py -v

GPU 없이 실행 가능. 실제 모델 로드 없이 로직만 검증합니다.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
import torch

# 프로젝트 루트를 sys.path에 추가
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 1. get_device() — 환경별 디바이스 자동 감지
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        from src.utils.device import get_device
        d = get_device()
        assert isinstance(d, torch.device)

    def test_device_is_valid_type(self):
        from src.utils.device import get_device
        d = get_device()
        assert d.type in ("cuda", "mps", "cpu")

    def test_cuda_if_available(self):
        from src.utils.device import get_device
        d = get_device()
        if torch.cuda.is_available():
            assert d.type == "cuda"

    def test_mps_if_cuda_unavailable(self):
        from src.utils.device import get_device
        d = get_device()
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            assert d.type == "mps"

    def test_cpu_fallback(self):
        from src.utils.device import get_device
        d = get_device()
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            assert d.type == "cpu"


# ---------------------------------------------------------------------------
# 2. Preprocess.make_input() — train 모드 / test 모드
# ---------------------------------------------------------------------------

class TestPreprocess:
    @pytest.fixture
    def preprocessor(self):
        from src.data.preprocess import Preprocess
        return Preprocess(bos_token="<s>", eos_token="</s>")

    @pytest.fixture
    def sample_train_df(self):
        return pd.DataFrame({
            "fname": ["train_0", "train_1"],
            "dialogue": ["#Person1#: 안녕하세요.\n#Person2#: 반갑습니다.", "#Person1#: 좋은 아침이에요."],
            "summary": ["두 사람이 인사를 나눴다.", "아침 인사를 했다."],
        })

    @pytest.fixture
    def sample_test_df(self):
        return pd.DataFrame({
            "fname": ["test_0", "test_1"],
            "dialogue": ["#Person1#: 감사합니다.", "#Person1#: 실례합니다."],
        })

    def test_train_mode_returns_three_tuples(self, preprocessor, sample_train_df):
        result = preprocessor.make_input(sample_train_df)
        assert len(result) == 3, "train 모드는 (encoder, decoder_in, decoder_out) 3개 반환"

    def test_train_mode_lengths_equal(self, preprocessor, sample_train_df):
        enc, dec_in, dec_out = preprocessor.make_input(sample_train_df)
        assert len(enc) == len(dec_in) == len(dec_out) == len(sample_train_df)

    def test_train_mode_bos_eos(self, preprocessor, sample_train_df):
        _, dec_in, dec_out = preprocessor.make_input(sample_train_df)
        for d in dec_in:
            assert d.startswith("<s>"), "decoder_input은 BOS로 시작해야 함"
        for d in dec_out:
            assert d.endswith("</s>"), "decoder_output은 EOS로 끝나야 함"

    def test_test_mode_returns_two_tuples(self, preprocessor, sample_test_df):
        result = preprocessor.make_input(sample_test_df, is_test=True)
        assert len(result) == 2, "test 모드는 (encoder, decoder_in) 2개 반환"

    def test_test_mode_decoder_all_bos(self, preprocessor, sample_test_df):
        _, dec_in = preprocessor.make_input(sample_test_df, is_test=True)
        for d in dec_in:
            assert d == "<s>", "test 모드 decoder_in은 모두 BOS여야 함"

    def test_prefix_applied(self, preprocessor, sample_train_df):
        enc, _, _ = preprocessor.make_input(sample_train_df, prefix="summarize: ")
        for e in enc:
            assert e.startswith("summarize: "), "prefix가 encoder 입력 앞에 붙어야 함"


# ---------------------------------------------------------------------------
# 3. clean_text() — 단독 자음/모음·빈 괄호·반복 특수기호 제거
# ---------------------------------------------------------------------------

class TestCleanText:
    def setup_method(self):
        from src.data.preprocess import clean_text
        self.clean = clean_text

    def test_removes_standalone_consonants(self):
        result = self.clean("ㅋㅋㅋ 안녕하세요")
        assert "ㅋ" not in result
        assert "안녕하세요" in result

    def test_removes_empty_parentheses(self):
        result = self.clean("좋아요 () 맞아요")
        assert "()" not in result

    def test_removes_repeated_special_chars(self):
        result = self.clean("정말!!!!!!!")
        assert "!!!!!!!" not in result

    def test_cleans_multiple_spaces(self):
        result = self.clean("안녕    하세요")
        assert "    " not in result

    def test_preserves_person_tags(self):
        text = "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
        result = self.clean(text)
        assert "#Person1#" in result
        assert "#Person2#" in result

    def test_preserves_masking_tokens(self):
        text = "#PhoneNumber# 으로 전화해주세요."
        result = self.clean(text)
        assert "#PhoneNumber#" in result

    def test_no_change_for_clean_text(self):
        text = "두 사람이 약속을 정했다."
        result = self.clean(text)
        assert result == text


# ---------------------------------------------------------------------------
# 4. filter_by_length() — 길이 기반 이상치 필터
# ---------------------------------------------------------------------------

class TestFilterByLength:
    def setup_method(self):
        from src.data.preprocess import filter_by_length
        self.filter = filter_by_length

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "fname": ["a", "b", "c"],
            "dialogue": ["정상 대화", "x" * 2000, "또 다른 대화"],
            "summary": ["정상 요약 텍스트입니다.", "요약", "정상 요약 텍스트입니다."],
        })

    def test_removes_long_dialogue(self, df):
        result = self.filter(df, dialogue_max=1500, summary_min=5, summary_max=250)
        assert all(result["dialogue"].str.len() <= 1500)

    def test_removes_short_summary(self, df):
        result = self.filter(df, dialogue_max=1500, summary_min=10, summary_max=250)
        assert all(result["summary"].str.len() >= 10)

    def test_resets_index(self, df):
        result = self.filter(df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# 5. postprocess() — 5단계 후처리
# ---------------------------------------------------------------------------

class TestPostprocess:
    def setup_method(self):
        from src.utils.postprocess import postprocess
        self.pp = postprocess

    def test_removes_special_tokens(self):
        text = "<s>요약 텍스트입니다.</s>"
        result = self.pp(text, remove_tokens=["<s>", "</s>"])
        assert "<s>" not in result
        assert "</s>" not in result

    def test_ensures_trailing_period(self):
        result = self.pp("요약 텍스트")
        assert result.endswith(".")

    def test_no_extra_period_if_already_ends_with_period(self):
        result = self.pp("요약 텍스트.")
        assert result.count("..") == 0

    def test_removes_repeated_sentences(self):
        text = "첫 번째 문장. 두 번째 문장. 첫 번째 문장."
        result = self.pp(text)
        assert result.count("첫 번째 문장") == 1

    def test_cleans_whitespace(self):
        result = self.pp("요약   텍스트")
        assert "   " not in result

    def test_min_length_flag_via_batch(self):
        from src.utils.postprocess import batch_postprocess_with_flags
        texts = ["짧", "충분히 긴 요약 텍스트입니다."]
        _, flags = batch_postprocess_with_flags(texts, min_length=10)
        assert flags[0] is True
        assert flags[1] is False


# ---------------------------------------------------------------------------
# 6. compute_metrics() — ROUGE 계산 (동일 문장 → ROUGE=1.0)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_score_baseline_mode(self):
        from src.utils.metrics import _rouge_baseline
        preds = ["두 사람이 인사를 나눴다."]
        golds = ["두 사람이 인사를 나눴다."]
        scores = _rouge_baseline(preds, golds)
        assert abs(scores["rouge-1"] - 1.0) < 1e-5
        assert abs(scores["rouge-2"] - 1.0) < 1e-5
        assert abs(scores["rouge-l"] - 1.0) < 1e-5

    def test_zero_score_for_different_text(self):
        from src.utils.metrics import _rouge_baseline
        preds = ["가나다라마바사"]
        golds = ["ABCDEFGHIJK"]
        scores = _rouge_baseline(preds, golds)
        assert scores["rouge-1"] == 0.0

    def test_rouge_combined_is_sum(self):
        from src.utils.metrics import _rouge_baseline
        preds = ["두 사람이 인사를 나눴다."]
        golds = ["두 사람이 인사를 나눴다."]
        scores = _rouge_baseline(preds, golds)
        combined = scores["rouge-1"] + scores["rouge-2"] + scores["rouge-l"]
        assert abs(combined - 3.0) < 1e-4

    def test_multi_ref_rouge(self):
        from src.utils.metrics import compute_multi_ref_rouge
        pred = "두 사람이 만났다."
        refs = ["두 사람이 만났다.", "두 사람이 만남을 가졌다.", "두 명이 만났다."]
        result = compute_multi_ref_rouge(pred, refs)
        assert "rouge_1_f1" in result
        assert "rouge_combined" in result
        assert result["rouge_1_f1"] > 0.0


# ---------------------------------------------------------------------------
# 7. DatasetForSeq2Seq / DatasetForInference
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_seq2seq_dataset_len(self):
        from src.data.preprocess import DatasetForSeq2Seq
        from unittest.mock import MagicMock
        n = 5
        enc = MagicMock()
        enc.__iter__ = MagicMock(return_value=iter([]))
        enc.items = MagicMock(return_value=[("input_ids", torch.zeros(n, 10, dtype=torch.long))])
        enc.__getitem__ = MagicMock(return_value=torch.zeros(10, dtype=torch.long))
        enc["input_ids"] = torch.zeros(n, 10, dtype=torch.long)

        dec_in = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        labels = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        enc_dict = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}

        ds = DatasetForSeq2Seq(enc_dict, dec_in, labels)
        assert len(ds) == n

    def test_seq2seq_dataset_getitem_keys(self):
        from src.data.preprocess import DatasetForSeq2Seq
        n = 3
        enc = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        dec = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        lbl = {"input_ids": torch.zeros(n, 10, dtype=torch.long), "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        ds = DatasetForSeq2Seq(enc, dec, lbl)
        item = ds[0]
        assert "input_ids" in item
        assert "decoder_input_ids" in item
        assert "labels" in item


# ---------------------------------------------------------------------------
# 8. compare_rouge_modes() — baseline vs korouge 비교
# ---------------------------------------------------------------------------

class TestCompareRougeModes:
    def test_returns_both_modes(self):
        from src.utils.metrics import compare_rouge_modes
        preds = ["두 사람이 인사를 나눴다."]
        refs = ["두 사람이 인사를 나눴다."]
        result = compare_rouge_modes(preds, refs)
        assert "baseline" in result
        assert "korouge" in result

    def test_scores_are_floats(self):
        from src.utils.metrics import compare_rouge_modes
        preds = ["요약 텍스트."]
        refs = ["요약 텍스트."]
        result = compare_rouge_modes(preds, refs)
        for mode in ("baseline", "korouge"):
            for key, val in result[mode].items():
                assert isinstance(val, float), f"{mode}/{key} 는 float이어야 함"


# ---------------------------------------------------------------------------
# 9. reverse_utterances() / apply_tta()
# ---------------------------------------------------------------------------

class TestTTA:
    def test_reverse_utterances(self):
        from src.data.preprocess import reverse_utterances
        dialogue = "#Person1#: 안녕.\n#Person2#: 반가워."
        result = reverse_utterances(dialogue)
        lines = result.split("\n")
        assert lines[0].startswith("#Person2#")
        assert lines[1].startswith("#Person1#")

    def test_apply_tta_2way(self):
        from src.data.preprocess import apply_tta
        dialogues = ["#Person1#: A.\n#Person2#: B.", "#Person1#: X.\n#Person2#: Y."]
        variants = apply_tta(dialogues, n_ways=2)
        assert len(variants) == 2
        assert len(variants[0]) == 2  # 원본 + 역전
        assert variants[0][0] != variants[0][1]

    def test_apply_tta_1way_is_original(self):
        from src.data.preprocess import apply_tta
        dialogues = ["#Person1#: 안녕."]
        variants = apply_tta(dialogues, n_ways=1)
        assert variants[0][0] == dialogues[0]


# ---------------------------------------------------------------------------
# 10. MBRDecoder
# ---------------------------------------------------------------------------

class TestMBRDecoder:
    def test_returns_string(self):
        from src.ensemble import MBRDecoder
        decoder = MBRDecoder()
        candidates = ["첫 번째 요약.", "두 번째 요약.", "세 번째 요약."]
        result = decoder.decode(candidates)
        assert isinstance(result, str)
        assert result in candidates

    def test_empty_candidates(self):
        from src.ensemble import MBRDecoder
        decoder = MBRDecoder()
        result = decoder.decode([])
        assert result == ""

    def test_single_candidate(self):
        from src.ensemble import MBRDecoder
        decoder = MBRDecoder()
        result = decoder.decode(["유일한 요약."])
        assert result == "유일한 요약."


# ---------------------------------------------------------------------------
# 11. evaluate_multi_ref() — 대회 공식 채점 방식
# ---------------------------------------------------------------------------

class TestEvaluateMultiRef:
    def test_single_ref_same_as_standard(self):
        from src.utils.metrics import evaluate_multi_ref, _rouge_baseline
        preds = ["두 사람이 만났다."]
        refs = [["두 사람이 만났다."]]
        result = evaluate_multi_ref(preds, refs)
        std = _rouge_baseline(preds, ["두 사람이 만났다."])
        assert abs(result["rouge_1_f1"] - std["rouge-1"]) < 1e-5

    def test_multi_ref_score_in_range(self):
        from src.utils.metrics import evaluate_multi_ref
        preds = ["두 사람이 약속을 잡았다.", "날씨가 맑다."]
        refs = [
            ["두 사람이 약속을 잡았다.", "두 명이 만남을 정했다.", "약속이 잡혔다."],
            ["날씨가 맑다.", "맑은 날씨."],
        ]
        result = evaluate_multi_ref(preds, refs)
        assert 0.0 <= result["rouge_combined"] <= 3.0
