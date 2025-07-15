# GFlowNet Trajectory Uniqueness Analysis

## 파일 구성

### 분석 코드
- `correct_uniqueness_plot.py` - GFlowNet trajectory uniqueness 분석 및 시각화 코드

### 시각화 결과 (4개 개별 플롯)
- `plot1_unique_growth.png` - Unique Sequence Growth Over Time
- `plot2_uniqueness_rate.png` - Uniqueness Rate Over Time  
- `plot3_loglog_pattern.png` - Log-Log Scale Growth Pattern
- `plot4_final_comparison.png` - Final Uniqueness Rate by Problem

## 실행 방법

```bash
cd /home/ubuntu/GFN_to_ARC/gfn/analysis/
python correct_uniqueness_plot.py
```

## 분석 결과 요약

**분석 대상**: 7개 ARC 문제 (86, 139, 149, 154, 178, 240, 379)

**주요 발견사항**:
- 모든 문제에서 unique action sequence가 단 1개만 존재
- Uniqueness rate: 0.0% - 0.1% (극도로 낮음)
- GFlowNet 모델의 exploration 다양성 부족 확인

**데이터 위치**: `/data/gflownet-llm/`