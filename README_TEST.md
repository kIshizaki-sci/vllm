# Qwen2.5-VL Vision Component Dtype Test

## 概要
このテストスクリプトは、Qwen2.5-VLモデルのvisionコンポーネントが、GPUのbfloat16サポート状況に応じて適切なdtypeを選択することを検証します。

## テストの実行方法

### 方法1: Bashスクリプトを使用（推奨）
```bash
# 実行権限を付与
chmod +x run_qwen2_5_vl_test.sh

# テストを実行
./run_qwen2_5_vl_test.sh
```

### 方法2: 環境変数を手動設定して実行
```bash
# 必要な環境変数を設定
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0

# テストを実行
python test_qwen2_5_vl_dtype.py
```

### 方法3: ワンライナーで実行
```bash
MASTER_ADDR=localhost MASTER_PORT=12355 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python test_qwen2_5_vl_dtype.py
```

## テスト内容

### Test 1: GPU without bfloat16 support
- GPUがbfloat16をサポートしていない場合のシミュレーション
- 全visionコンポーネントがfloat32で初期化されることを確認

### Test 2: GPU with bfloat16 support  
- GPUがbfloat16をサポートしている場合のシミュレーション
- 全visionコンポーネントがbfloat16で初期化されることを確認

### Test 3: Weight Conversion
- bfloat16の重みをロードする際、float32への自動変換を確認

### Test 4: Merger Component
- PatchMergerコンポーネントのdtype処理を確認

## 期待される出力

成功時の出力例：
```
Testing Qwen2.5-VL Vision Component Dtype Handling
============================================================

Test 1: GPU without bfloat16 support
----------------------------------------
✓ Vision transformer dtype: torch.float32
✓ Patch embed weight dtype: torch.float32
✓ Block MLP gate_up_proj weight dtype: torch.float32
✓ Block Attention QKV weight dtype: torch.float32
✓ Merger MLP fc1 weight dtype: torch.float32
✓ Merger MLP fc2 weight dtype: torch.float32

Test 1: PASSED ✓

Test 2: GPU with bfloat16 support
----------------------------------------
✓ Vision transformer dtype: torch.bfloat16
✓ Patch embed weight dtype: torch.bfloat16

Test 2: PASSED ✓

============================================================
ALL TESTS COMPLETED SUCCESSFULLY! ✓✓✓
============================================================
```

## トラブルシューティング

### エラー: "tensor model parallel group is not initialized"
環境変数が正しく設定されていません。上記の実行方法を確認してください。

### エラー: "MASTER_ADDR expected, but not set"
`MASTER_ADDR`環境変数が設定されていません。`export MASTER_ADDR=localhost`を実行してください。

### エラー: ImportError
vLLMが正しくインストールされていることを確認してください：
```bash
pip install vllm
```

## 実装の詳細

修正されたファイル：
- `vllm/model_executor/models/qwen2_5_vl.py`: Vision transformerのdtype処理を追加

主な変更点：
1. GPU compute capabilityの自動検出
2. bfloat16非対応時のfloat32フォールバック
3. 全visionコンポーネントへのparams_dtype伝播
4. 重みロード時の自動dtype変換

## サポートされるGPU

| GPU | Compute Capability | 使用dtype |
|-----|-------------------|-----------|
| A100/H100 | >= 8.0 | bfloat16 |
| RTX 4090 | >= 8.0 | bfloat16 |
| RTX 3090 | 8.6 | bfloat16 |
| RTX 2080 Ti | 7.5 | float32 |
| GTX 1080 Ti | 6.1 | float32 |
| T4 | 7.5 | float32 |
