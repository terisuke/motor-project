#!/bin/bash

# レーン追従システム スタートアップスクリプト
# 実行: ./start_lane_follower.sh [オプション]

# 実行ディレクトリをスクリプトの場所に設定
cd "$(dirname "$0")"

# デフォルト設定
CAMERA=0
MOTORS_ENABLED=true
SPEED=0.5
COLOR_MARKERS=false
DEBUG=false
RECORD=false
LOG_DATA=false

# バナー表示
echo "====================================================="
echo "           レーン追従システム スタートアップ          "
echo "====================================================="

# 引数の解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      # テストモード（モーターなし、デバッグあり）
      MOTORS_ENABLED=false
      DEBUG=true
      shift
      ;;
    --no-motors)
      # モーターを無効化
      MOTORS_ENABLED=false
      shift
      ;;
    --speed=*)
      # 速度設定
      SPEED="${1#*=}"
      shift
      ;;
    --camera=*)
      # カメラ設定
      CAMERA="${1#*=}"
      shift
      ;;
    --color-markers)
      # 色マーカー検出を有効化
      COLOR_MARKERS=true
      shift
      ;;
    --record)
      # 録画モードを有効化
      RECORD=true
      shift
      ;;
    --debug)
      # デバッグモードを有効化
      DEBUG=true
      shift
      ;;
    --log-data)
      # データログを有効化
      LOG_DATA=true
      shift
      ;;
    --help)
      # ヘルプ表示
      echo "使用方法: $0 [オプション]"
      echo "オプション:"
      echo "  --test          テストモード（モーターなし、デバッグあり）"
      echo "  --no-motors     モーター制御を無効化"
      echo "  --speed=X       モーター速度を設定（0.0-1.0）"
      echo "  --camera=X      カメラデバイスIDを設定"
      echo "  --color-markers 色マーカー検出を有効化"
      echo "  --record        走行録画を有効化"
      echo "  --debug         デバッグ情報を表示"
      echo "  --log-data      データログ記録を有効化"
      echo "  --help          このヘルプメッセージを表示"
      echo ""
      echo "プリセット:"
      echo "  ./start_lane_follower.sh --test            # テストモード"
      echo "  ./start_lane_follower.sh --color-markers   # 色マーカー検出有効"
      echo "  ./start_lane_follower.sh --speed=0.7       # 高速モード"
      exit 0
      ;;
    *)
      echo "不明なオプション: $1"
      echo "ヘルプを表示するには --help を使用してください"
      exit 1
      ;;
  esac
done

# 現在の設定を表示
echo "システム設定:"
echo "- カメラ: $CAMERA"
echo "- モーター: $([ "$MOTORS_ENABLED" = true ] && echo "有効" || echo "無効")"
echo "- 速度: $SPEED"
echo "- 色マーカー検出: $([ "$COLOR_MARKERS" = true ] && echo "有効" || echo "無効")"
echo "- デバッグモード: $([ "$DEBUG" = true ] && echo "有効" || echo "無効")"
echo "- 録画: $([ "$RECORD" = true ] && echo "有効" || echo "無効")"
echo "- データログ: $([ "$LOG_DATA" = true ] && echo "有効" || echo "無効")"

# コマンドライン引数を構築
CMD="python3 src/lane_following/improved_integrated_lane_follower.py --camera $CAMERA"

# オプションの追加
[ "$MOTORS_ENABLED" = false ] && CMD="$CMD --no-motors"
[ "$COLOR_MARKERS" = true ] && CMD="$CMD --color-markers"
[ "$DEBUG" = true ] && CMD="$CMD --debug"
[ "$RECORD" = true ] && CMD="$CMD --record"
[ "$LOG_DATA" = true ] && CMD="$CMD --log-data"
CMD="$CMD --speed $SPEED"

echo ""
echo "実行コマンド: $CMD"
echo ""
echo "システムを開始します... (終了するには Ctrl+C)"
echo "====================================================="

# システムの実行
$CMD

# 終了コードの確認
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "システムはエラーコード $EXIT_CODE で終了しました。"
else
  echo "システムは正常に終了しました。"
fi

exit $EXIT_CODE