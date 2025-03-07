#!/bin/bash

# レーン追従システム スタートアップスクリプト
# 使用方法: ./start_lane_follower.sh [オプション]

# デフォルト設定
CAMERA=0
NO_MOTORS=false
SPEED=0.5
MAX_STEERING=0.5
STEERING_SENSITIVITY=1.0
COLOR_MARKERS=false
RECORD=false
DEBUG=false
LOG_DATA=false
TEST_MODE=false
HEADLESS=false

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --max-steering)
            MAX_STEERING="$2"
            shift 2
            ;;
        --steering-sensitivity)
            STEERING_SENSITIVITY="$2"
            shift 2
            ;;
        --color-markers)
            COLOR_MARKERS=true
            shift
            ;;
        --record)
            RECORD=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --log-data)
            LOG_DATA=true
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --test)
            TEST_MODE=true
            NO_MOTORS=true
            DEBUG=true
            shift
            ;;
        --help)
            echo "使用方法: ./start_lane_follower.sh [オプション]"
            echo "オプション:"
            echo "  --camera N              カメラデバイスのインデックス（デフォルト: 0）"
            echo "  --speed N               基本速度 0-1（デフォルト: 0.5）"
            echo "  --max-steering N        最大ステアリング量（デフォルト: 0.5）"
            echo "  --steering-sensitivity N ステアリング感度（デフォルト: 1.0）"
            echo "  --color-markers         カラーマーカー検出を有効化"
            echo "  --record                走行映像の録画を有効化"
            echo "  --debug                 デバッグ情報の表示"
            echo "  --log-data              データログ記録を有効化"
            echo "  --headless              ヘッドレスモード（GUI表示なし）"
            echo "  --test                  テストモード（モーター無効、デバッグ有効）"
            echo "  --help                  このヘルプメッセージを表示"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            echo "ヘルプを表示するには: ./start_lane_follower.sh --help"
            exit 1
            ;;
    esac
done

# コマンドの構築
CMD="python3 src/lane_following/improved_integrated_lane_follower.py"
CMD="$CMD --camera $CAMERA"

if [ "$NO_MOTORS" = true ]; then
    CMD="$CMD --no-motors"
fi

CMD="$CMD --speed $SPEED"
CMD="$CMD --max-steering $MAX_STEERING"
CMD="$CMD --steering-sensitivity $STEERING_SENSITIVITY"

if [ "$COLOR_MARKERS" = true ]; then
    CMD="$CMD --color-markers"
fi

if [ "$RECORD" = true ]; then
    CMD="$CMD --record"
fi

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

if [ "$LOG_DATA" = true ]; then
    CMD="$CMD --log-data"
fi

if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

# 設定情報の表示
echo "====================================================="
echo "           レーン追従システム スタートアップ          "
echo "====================================================="
echo "システム設定:"
echo "- カメラ: $CAMERA"
echo "- モーター: $([ "$NO_MOTORS" = true ] && echo "無効" || echo "有効")"
echo "- 速度: $SPEED"
echo "- 色マーカー検出: $([ "$COLOR_MARKERS" = true ] && echo "有効" || echo "無効")"
echo "- デバッグモード: $([ "$DEBUG" = true ] && echo "有効" || echo "無効")"
echo "- 録画: $([ "$RECORD" = true ] && echo "有効" || echo "無効")"
echo "- データログ: $([ "$LOG_DATA" = true ] && echo "有効" || echo "無効")"
echo "- ヘッドレスモード: $([ "$HEADLESS" = true ] && echo "有効" || echo "無効")"
echo ""
echo "実行コマンド: $CMD"
echo ""
echo "システムを開始します... (終了するには Ctrl+C)"
echo "====================================================="

# コマンドの実行
$CMD
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "システムはエラーコード $EXIT_CODE で終了しました。"
    exit $EXIT_CODE
fi