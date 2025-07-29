# GPT from Scratch - M2チップ最適化版

このプロジェクトは、Andrej Karpathyの「Let's build GPT: from scratch, in code, spelled out.」に基づいて、MacBook M2チップ環境に最適化したGPT実装です。

## 特徴

- **M2チップ最適化**: Metal Performance Shaders (MPS) を使用した高速化
- **メモリ効率**: 16GB MacBookに最適化されたハイパーパラメータ
- **エラーハンドリング**: 堅牢なエラー処理とデバッグ情報
- **チェックポイント機能**: 訓練途中のモデル保存と復元

## 環境要件

- macOS (M2チップ推奨)
- Python 3.8+
- PyTorch 2.0+ (MPS対応版)

## インストール

```bash
pip install torch torchvision torchaudio
```

## 使用方法

1. **データ準備**: `input.txt` ファイルをプロジェクトルートに配置
2. **実行**: 
   ```bash
   python buildGPTfromscratch_4_bk.py
   ```

## ハイパーパラメータ

- `batch_size`: 64 (メモリ効率のため)
- `block_size`: 128 (コンテキスト長)
- `n_embd`: 128 (埋め込み次元)
- `n_head`: 6 (アテンションヘッド数)
- `n_layer`: 6 (Transformer層数)
- `learning_rate`: 3e-4

## ファイル構成

- `buildGPTfromscratch_4_bk.py`: メインのGPT実装
- `input.txt`: 訓練用テキストデータ
- `checkpoints/`: モデルチェックポイント保存ディレクトリ

## 出力

- 訓練中の損失値表示
- チェックポイントファイル（`checkpoints/`ディレクトリ）
- 生成されたテキストサンプル

## ライセンス

MIT License

## 参考

- [Andrej Karpathy's GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html) 