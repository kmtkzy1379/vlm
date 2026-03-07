# VLM Screen Recognition - Brain-Inspired Enhancement Plan

## 設計原理: 人間の視覚野に基づくパイプライン

```
人間の視覚野          本システムの対応モジュール
─────────────────    ─────────────────────────
V1 (エッジ検出)   →  SaliencyDetector (顕著性マップ)
V2 (テクスチャ)   →  PredictiveCoder (予測誤差のみ処理)
V4 (形状・色)     →  HierarchicalAnalyzer (マルチスケール特徴)
IT (物体認識)     →  YOLODetector + PerIDAnalyzer (物体認識)
MT/V5 (動き)      →  OpticalFlowMotion (オプティカルフロー)
前頭前野 (注意)   →  AttentionWeighter (顕著性重み付き処理)
海馬 (記憶)       →  WorkingMemory (物体永続性 + シーングラフ)
```

## タスク一覧

### Phase A: 予測符号化 (Predictive Coding) - 脳のV2に相当
脳は「予測と現実の差分」のみを処理する。静的な画面の90%を無視し、変化した領域だけを分析する。

- [x] A1. PredictiveCoder クラス作成 (`src/vlm/capture/predictive_coder.py`)
  - フレーム差分による変化領域マスク生成
  - 変化領域の矩形リスト (ROI) 抽出
  - ChangeDetector と統合: 変化レベル判定 + 変化領域の特定
- [x] A2. テスト作成・実行 (`tests/test_capture/test_predictive_coder.py`) ✅ 10/10 passed

### Phase B: 顕著性検出 (Saliency Detection) - 脳のV1注意機構に相当
脳は視野全体を均等に処理せず、「目立つ領域」に注意を集中する。

- [x] B1. SaliencyDetector クラス作成 (`src/vlm/capture/saliency.py`)
  - 自前Spectral Residual実装 (opencv-contrib不要)
  - 顕著性スコアでROIに優先度を付与
  - PredictiveCoder と統合: 変化あり + 顕著 = 最優先
- [x] B2. テスト作成・実行 (`tests/test_capture/test_saliency.py`) ✅ 7/7 passed

### Phase C: オプティカルフロー動き検出 - 脳のMT/V5に相当
脳のMT野はピクセルレベルの動きを検出する。bbox中心移動より精密な動き理解。

- [x] C1. OpticalFlowMotion クラス作成 (`src/vlm/analysis/optical_flow.py`)
  - Farneback dense optical flow で動きベクトル場を計算
  - エンティティ領域内の平均フロー → 高精度な速度/方向
  - 既存 MotionDetector の上位互換として統合
- [x] C2. テスト作成・実行 (`tests/test_analysis/test_optical_flow.py`) ✅ 7/7 passed

### Phase D: シーングラフ (Scene Graph) - 脳の空間認知に相当
脳は物体を孤立して認識せず、空間関係 (上・下・内・隣接) を理解する。

- [x] D1. SceneGraphBuilder クラス作成 (`src/vlm/aggregation/scene_graph.py`)
  - 純Python実装 (NetworkX不要、軽量)
  - エンティティ間の空間関係自動推論 (above, below, left_of, right_of, inside, contains, overlapping, near)
  - delta mode: 変化した関係のみ出力
- [x] D2. テスト作成・実行 (`tests/test_aggregation/test_scene_graph.py`) ✅ 11/11 passed

### Phase E: ワーキングメモリ (Working Memory) - 脳の海馬に相当
脳は一時的に見えなくなった物体を「忘れない」。物体永続性 + エピソード記憶。

- [x] E1. WorkingMemory クラス作成 (`src/vlm/tracking/working_memory.py`)
  - 消失エンティティの色ヒストグラム外見特徴を保持
  - 再出現時のReID (ヒストグラム相関マッチング)
  - エピソード記憶: 重要イベントの時系列ログ
- [x] E2. テスト作成・実行 (`tests/test_tracking/test_working_memory.py`) ✅ 10/10 passed

### Phase F: パイプライン統合 + main.py 更新
全モジュールをパイプラインに統合し、設定で有効/無効を切替可能にする。

- [x] F1. main.py を更新: 全新モジュールを統合 ✅
  - PredictiveCoder → SaliencyDetector → Detection → Tracking → Analysis → SceneGraph → WorkingMemory → DeltaEncoder → LLM
- [x] F2. config/default.yaml に新設定項目追加 ✅
- [x] F3. 全テスト実行・統合確認 ✅ 73/73 passed

### Phase G: LLMプロンプト強化
シーングラフ + 空間関係 + ワーキングメモリ情報をLLMに提供。

- [x] G1. prompt_builder.py 更新: シーングラフ・空間関係をコンパクト形式で含める ✅
- [x] G2. テスト作成・実行 ✅ 8/8 passed (全体: 81/81 passed)
