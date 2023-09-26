# uec20_risaju_backwend
## 自分用コマンドまとめ
- docker内でのinstallされたモジュール確認方法
```bash
pip freeze | grep <モジュール名>
```
- dockerのコンテナでマウントしたフォルダへのuser権限付与
```bash
sudo chown -R $USER:$USER ./<マウントしたフォルダ名>
```
- gitignoreが反映されない場合
```bash
git rm -r --cached [ファイル名]
```
## Issue
- WSLを使用しているとpyaudioが使用できない問題\
WSL上では録音するためのデバイスを認識できないためエラーが出る
おそらくLinuxで録音するためのデバイスを設定することが必要

## backendServerの起動方法
- dockerの起動
```bash
docker-compose up -d
```
- dockerコンテナに入る
```bash
docker-compose exec app bash
```
- fastAPIサーバーの起動
    - vr_voice_chatフォルダ内に.envファイルを作成
    - .envファイルに以下を記述
        ```bash
        OPENAI_API_KEY=<openaiのAPIキー>
        ```
    - emotion-analysis-modelをダウンロード
    - vr_voice_chatフォルダ内にemotion-analysis-modelフォルダを配置
    - dockerコンテナ内で
        ```bash
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```
    - 以下にアクセスしてHello Worldが表示されれば成功
        http://localhost:8080/
- APIの確認
    - 以下にアクセスしてAPIの仕様が表示されれば成功
http://localhost:8080/docs
## チャットと感情分析APIの使用方法
- APIのエンドポイントは"/api/v1/chat"のpost通信
- パラメータの意味
    - APIパラメータ
        - text (str): userの発言
        - emotion (bool) : 感情分析を行うかどうか
    - EmotionAnalysisクラスのパラメータ
        - use_japanese (bool): 感情のラベル日本語かするかどうか
        - use_gpu (bool) : GPUを使用するかどうか
    - EmotionAnalysis.analyze_emotion()のパラメータ
        - show_fig : matplotlibによる表の出力を行うかどうか
        - ret_prob : 感情の分析の結果を返すかどうか
- APIの返り値
    - response (str) : chatbotの返答
    - emotion (str) : 感情分析の結果\
    英語['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']\
    日本語['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
    - strength (float64) : 感情の強さ\
    0(弱い)~1(強い)
