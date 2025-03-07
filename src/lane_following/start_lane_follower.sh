#!/bin/bash

# レーン追従システム起動スクリプト

# スクリプトが存在するディレクトリへ移動
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 環境変数のセットアップ
export PYTHONPATH="$SCRIPT_DIR/../..:$PYTHONPATH"

# 仮想環境のPythonを直接指定
PYTHON_PATH="/home/terisuke/develop/motor_project/env/bin/python3"

# 引数の処理
CAMERA=0
DEBUG=""
NO_MOTORS=""
COLOR_MARKERS=""
RECORD=""
HEADLESS="--headless"  # デフォルトでヘッドレスモードを有効に
HELP_MSG="""
使用法: $0 [オプション]

オプション:
  -c, --camera N      カメラインデックス番号 (デフォルト: 0)
  -d, --debug         デバッグモードを有効化
  -n, --no-motors     モーター制御を無効化（シミュレーションモード）
  -m, --markers       カラーマーカー検出を有効化
  -r, --record        映像録画を有効化
  -g, --gui           GUIモードを有効化（デフォルトはヘッドレスモード）
  -h, --help          このヘルプメッセージを表示
"""

while [ "$1" != "" ]; do
    case $1 in
        -c | --camera)
            shift
            CAMERA=$1
            ;;
        -d | --debug)
            DEBUG="--debug"
            ;;
        -n | --no-motors)
            NO_MOTORS="--no-motors"
            ;;
        -m | --markers)
            COLOR_MARKERS="--color-markers"
            ;;
        -r | --record)
            RECORD="--record"
            ;;
        -g | --gui)
            HEADLESS=""  # ヘッドレスモードを無効化
            ;;
        -h | --help)
            echo "$HELP_MSG"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            echo "$HELP_MSG"
            exit 1
            ;;
    esac
    shift
done

echo "レーン追従システムを起動します..."
echo "カメラインデックス: $CAMERA"
if [ "$DEBUG" != "" ]; then echo "デバッグモード: 有効"; fi
if [ "$NO_MOTORS" != "" ]; then echo "モーター制御: 無効"; fi
if [ "$COLOR_MARKERS" != "" ]; then echo "カラーマーカー検出: 有効"; fi
if [ "$RECORD" != "" ]; then echo "録画: 有効"; fi
if [ "$HEADLESS" != "" ]; then echo "ヘッドレスモード: 有効"; fi

# 依存関係の確認
echo "依存関係を確認中..."
REQUIRED_PACKAGES=("cv2" "numpy" "gpiozero")
MISSING=0

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    $PYTHON_PATH -c "import $pkg" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "エラー: Python パッケージ $pkg が見つかりません"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "必要なパッケージが不足しています。以下のコマンドでインストールしてください:"
    echo "pip install opencv-python numpy gpiozero"
    exit 1
fi

# 改良版 or 標準版
echo "改良版レーン追従システムを実行します..."
$PYTHON_PATH "$SCRIPT_DIR/improved_integrated_lane_follower.py" \
    --camera $CAMERA $DEBUG $NO_MOTORS $COLOR_MARKERS $RECORD $HEADLESS

# 終了コードの確認
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "エラー: レーン追従システムが終了コード $EXIT_CODE で終了しました"
    
    # 改良版が失敗した場合、標準版を試す
    echo "標準版でリトライします..."
    $PYTHON_PATH "$SCRIPT_DIR/integrated_lane_follower.py" \
        --camera $CAMERA $DEBUG $NO_MOTORS $COLOR_MARKERS $RECORD $HEADLESS
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "エラー: 標準版も終了コード $EXIT_CODE で終了しました"
    fi
fi

exit $EXIT_CODE