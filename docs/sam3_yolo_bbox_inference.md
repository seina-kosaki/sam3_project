# SAM3テキスト推論へのYOLO bbox入力

## 状態

- 合意済み要件。実装前仕様。
- 実装対象外: 従来の保存先、レビュー画面、承認・公開契約、学習用YOLOデータの形式。

## 目的

通常のSAM3テキスト推論に加え、設定が有効な場合だけYOLO検出の全bboxをSAM3のvisual promptとして渡し、同じテキストプロンプトとの組み合わせで推論する。

最終候補・草案・review・保存・承認・公開には、SAM3推論結果だけを使う。YOLOの検出矩形を最終成果物として保存したり、SAM3候補を置換したりしない。

## 入力

### 設定

`src/configs/capture.json` に次を追加する。

| キー | 型 | 初期値 | 意味 |
|---|---|---:|---|
| `use_yolo` | boolean | `false` | YOLO bboxをSAM3推論へ渡すか |
| `yolo_confidence` | number | `0.8` | YOLO検出のconfidence閾値 |

`yolo_confidence` は有限数かつ `0 <= value <= 1` を満たすこと。無効値は設定エラーとして推論を開始しない。既存設定にキーがない場合はこの初期値を適用する。

YOLO重み設定には `src/configs/settings_yolo.yaml` を使用する。本仕様で重みファイルのパスや学習設定を新設しない。YOLOは1クラスのみであるため、クラスIDによるフィルタを追加せず、閾値を通過した全検出を利用する。

### 推論入力

- 共通SAM3プロンプトは既存の `prompt` を使う。
- SAM3画像は `CaptureSample` の検証済みRGB画像を使う。
- YOLO bboxはピクセル座標の `[xmin, ymin, xmax, ymax]` とする。
- 有効bboxは画像範囲内に正規化する。座標値0は画像端を示す有効値として扱い、形式不正・非有限値・画像範囲へclipした後に幅または高さが0以下になるbboxのみ除外する。全てのbboxが無効になった場合は、有効検出0件としてテキストのみ推論へフォールバックする。

## 動作

1. 設定を読み、`use_yolo=false`なら既存のSAM3テキスト推論だけを行う。
2. `use_yolo=true`の場合、`src/configs/settings_yolo.yaml` の設定に基づいてYOLO推論を行う。
3. YOLO推論が正常終了し、有効bboxが1件以上ある場合、既存のSAM3テキストプロンプトと全bboxを同一画像のSAM3推論へ渡す。
4. YOLO推論が正常終了して有効bboxが0件の場合、bboxを渡さず、同じプロンプトでSAM3テキスト単独推論を行う。
5. YOLO推論自体が失敗した場合、SAM3テキスト単独推論へフォールバックしない。エラーを明示して当該推論を失敗扱いとする。
6. `use_yolo=false`の場合はYOLOをロード・実行しない。
7. SAM3の出力候補だけを既存のCandidate・bbox・score・maskへ変換する。
8. 既存のマッチング、草案、レビュー、承認、公開処理へSAM3候補だけを渡す。YOLO検出はレビュー対象や公開物へ直接混ぜない。

## 出力と保存

- SAM3候補が無ければ従来どおり空の候補列を保存し、推論完了を示す。
- 最終出力はSAM3のmask/bbox/scoreのみ。
- セッションDBの既存`sam3_predictions.model_metadata_json`へ、少なくとも次を記録する。
  - `use_yolo`
  - `yolo_confidence`
  - YOLO重み識別
  - 既存の `checkpoint`, `bpe_path`, `prompt`, `min_score`
- 重み識別には設定から解決したYOLO重みファイルの内容hashを使用する。
- SQLスキーマの追加・変更は当面不要とする。`sam3_predictions.model_signature`と既存`model_metadata_json`で記録できる。
- YOLO使用有無は、SQL保存時に必ず識別できる値として記録する。`model_metadata_json`が欠損またはnullではなく、YOLO使用時・不使用時の両方で明示する。

## キャッシュ署名

既存の画像hash + predictor.signature方式を維持し、`Sam3CapturePredictor.signature`へ次を含める。

- SAM3 `prompt`, `min_score`, checkpoint/BPE、SAM3実装・バージョン、torch
- YOLO使用有無
- YOLO confidence閾値
- YOLO重みファイルの内容hash
- YOLO推論実装の内容識別またはYOLOパッケージ版

`use_yolo=false`でも設定値を署名に含めてもよいが、有効・無効の切替は必ずsignatureとキャッシュキーを変える。`yolo_confidence`や `settings_yolo.yaml` から解決した重みファイルの内容hashが変わった場合は既存キャッシュを再利用しない。

`capture.json`を変更しても、設定値をpredictorへ注入しsignatureへ反映しない実装にはしない。予測完了行は異なるprediction_idを持つ必要がある。

## 制約

- YOLOは1クラス前提で全bboxをSAM3へ渡す。クラス名による選別は行わない。
- 変更範囲はcapture設定注入、predictor、YOLO利用箇所、既存signature・metadata記録へ限定する。
- `Sam3Engine.detect`の既存`gt_boxes`引数を使い、空リストでテキストのみの推論とする。
- 画像encodeは設定に関係なく従来どおり1回とし、プロンプトとbbox処理はencode後のstateで行う。
- 外部公開契約、`sam3_prediction_objects`スキーマ、`sam3_predictions`のテーブル・カラムは変更しない。
- YOLO/SAM3のプロンプト以外の推論条件を新設しない。

## 受け入れ条件

1. `use_yolo=false`またはキー省略時、現在のSAM3テキストのみ推論、結果、保存形式、キャッシュ動作を維持する。
2. `use_yolo=true`かつYOLOがbboxを返した場合、閾値を通過した全bboxと既存`prompt`がSAM3推論へ渡される。
3. `use_yolo=true`かつ正常検出が0件の場合、bboxなしでテキストのみSAM3推論を行い、空候補で終了せず既存推論結果を保存する。
4. `yolo_confidence`をcapture.jsonで設定でき、省略時は`0.8`になる。
5. 最終候補はSAM3結果だけであり、セッションSQLへYOLO使用有無が残る。
6. YOLO有効/無効、confidence、YOLO重み内容hash、SAM3設定のいずれかが変わると、異なるpredictor signatureとprediction_idになる。特に `settings_yolo.yaml` で指定された重みファイルの内容hashがsignature識別に反映される。
7. YOLO推論が失敗した場合、SAM3単独推論へ暗黙に切り替わらず、エラーを明示して失敗扱いになる。

## testsに含めること

- 無効設定時はYOLOを呼ばず既存SAM3のみ推論を行う。
- 有効設定時は全bboxとテキストが`detect(..., gt_boxes=...)`へ渡される。
- 正常な0件検出時は`gt_boxes=[]`またはbbox省略でテキストのみ推論される。
- YOLO推論失敗時はテキスト単独推論を呼ばず、明示的な失敗として扱う。
- `yolo_confidence`の既定値とcapture.json注入、不正値拒否を検証する。
- `settings_yolo.yaml`から解決された重みファイルの内容hashがsignatureに含まれ、重み内容の変更でsignature/prediction_idが変わることを検証する。
- YOLO/SAM設定の差分ごとに異なるsignature/prediction_idになる。
- SQLの`model_metadata_json`にYOLO使用有無と閾値が保存される。
- 最終候補がSAM3結果のみであることを検証する。

## 未解決事項

- なし。
