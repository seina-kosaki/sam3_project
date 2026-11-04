# SAM3 Capture Annotation

撮影APIのSQLiteメタデータとRGB画像を読み、SAM3の検出候補をタグUIDに対応付け、ブラウザーで一括レビューしてYOLO互換バッチとして公開するプロジェクトです。

## Setup

```powershell
uv sync
```

Windowsでは`pywin32>=312`をWindows専用依存として導入する。`arena-api`・`camlib`などlock外の手動導入パッケージがある既存環境では、通常の同期で除去されるため、実行・検証には`uv run --no-sync ...`を使う。

Hugging Faceの`facebook/sam3`モデルにアクセスできるアカウントを用意し、`src/.env`に`HF_TOKEN`を設定します。ローカル重みを使う場合は`src/configs/capture.json`の`checkpoint`にファイルパスを設定します。`null`ならSAM3モデルビルダーがHugging Faceから取得します。

設定ファイルの`capture_root`、`output_root`、`prompt`、`min_score`を確認する。相対パスは設定ファイルの位置が基準である。正式maskの保存先は各セッション直下の`masks`である。`mask_root`は旧設定との互換用で、新規PNGの保存先には使わない。

## Run

プロジェクトルートで一度のコマンドを実行します。動作確認時だけ`--limit`を付けます。

```powershell
uv run python src/capture_workflow.py run --limit 5
```

問題なければ全画像を処理します。

```powershell
uv run python src/capture_workflow.py run
```

推論後にローカルのレビュー画面が開きます。画像一覧には実画像、SAM3の矩形、SQLiteタグ点、信頼度が表示されます。低信頼度・タグ不一致・曖昧な候補を詳細表示して修正し、一覧で全画像を確認済みにした後、担当者とバッチ名を入力して一括公開します。画面は`127.0.0.1`だけで待ち受けます。

推論済み草案のレビューを再開する場合:

```powershell
uv run python src/capture_workflow.py review
```

信頼度の初期スコア境界は`src/configs/capture.json`の`review_confidence`で設定します。高信頼度でも自動承認はせず、一覧確認を必須とします。

「YOLO注釈」の確認モードでは、器具タグ（UID＋個体番号）とbboxを相互選択して対応を確認する。同じタグの点・bbox・右側の行は同じ色と番号で表示する。未対応候補は破線で表示し、対象タグへ明示的に採用する。重なったbboxはクリックで巡回でき、右側の一覧からも選択できる。候補が使えない場合はタグを選んで描画モードへ切り替え、ドラッグして手描きbboxを採用する。1タグにつき採用bboxは最大1つであり、置換には確認が必要である。修正保存後は再承認・再公開が必要である。手描きはYOLO専用であり、DINOの推論bbox・mask・物体対応は変更しない。

公開manifestは`output_root/exports/<session_id>/manifest.json`に作られます。レビュー画面ではセッションを選択して画像を確認し、そのセッションIDを公開フォルダー名として自動利用します。YOLO側の承認済みバッチreaderで検証できます。後段のデータセット作成・学習手順は[撮影データ連携手順](docs/capture_workflow.md)を参照してください。

各セッションの`capture_root/YYYY-MM-DD/<session_id>/metadata.sqlite3`が撮影・推論・レビュー・公開状態の正本である。ルート直下の旧単一DBは参照・生成しない。レビュー・承認・公開では画像本体を複製せず、共有ルート相対パスを参照する。

| 専用テーブル | 保存単位・内容 | 公開条件 |
|---|---|---|
| `sam3_dino_results` | レビュー版・物体番号ごとのscore、prompt、閾値、モデル情報、推論bbox、maskパス・hash、器具対応・採否履歴参照 | `is_dino_ready=1`：現行画像の承認＋物体別採用＋撮影情報の一致 |
| `sam3_yolo_results` | 画像・レビュー版ごとの修正済み注釈、原画像情報、承認・例外確認、manifest参照 | `is_yolo_ready=1`：現行承認に一致するmanifestの公開完了 |

両フラグは独立して管理する。取消・版更新・注釈編集で公開を失効させ、撮影行のタグ版・原本情報の変更や削除もSQLトリガーで即時に無効化する。DINO用bbox・maskはYOLO注釈を編集しても不変である。フラグは下流の処理済み状態ではなく、取得側でもファイルの存在・hashを検証する必要がある。

**今回の対応はSAM3側の保存までである。** DINOv3・YOLO側のセッションDB横断読み取りは別途対応が必要であり、現状の受け取りコードをそのまま新しい保存先へ接続できるとは限らない。YOLO互換manifest・注釈JSONの形式は維持する。

## 推論原本の保存と旧NPZ移行

`run`は推論bbox・score・モデル署名を所属セッションDBへ記録し、maskは確認用にメモリー上で保持する。承認前はPNGを保存しない。画像承認時に初めて`日付/<session_id>/masks/<画像名>_<物体番号>.png`へ保存する。PNGは元RGBと同寸法、背景0／物体255である。検出0件もSQLへ記録する。同一IDの異なる推論結果は上書きしない。

旧NPZが同じ推論キーで存在する場合は、`run`時に自動取り込みする。既存草案が参照するNPZをモデル実行なしでまとめて移行する場合は、**先に各セッションDBと`annotation_work`をバックアップ**してから実行する。旧単一DBからの移行は行わない。

```powershell
uv run python src/capture_workflow.py migrate-predictions
```

移行は再実行可能であり、NPZ・レビュー・承認・公開成果物は削除しません。既存のbbox承認をmask確認済みへ自動昇格させません。詳しい対象・制約は[撮影データ連携手順](docs/capture_workflow.md#推論原本のsqlpng保存と移行)を参照してください。

## 物体別DINO確認

各セッションDBと`annotation_work`をバックアップしてから、既存草案の確認画面を開く。`review`は追加SQL schemaと現行レビューを非破壊同期する。一覧表示だけではモデルやGPUを初期化しない。maskの確認時に保存済みデータがなければSAM3を遅延初期化して再推論する。再起動前と同じ推論設定・GPU環境が必要であり、設定や推論結果の不一致はエラーとして通知する。

保存済みmaskのファイル欠損は正常に再推論を待つ状態として扱い、画像ごとの理由とパスはDEBUGにのみ記録する。通常のINFOにはこの欠損通知を出さず、実際の再推論開始や他の警告・例外は従来どおり記録する。

```powershell
uv run python src/capture_workflow.py review
```

一覧の「DINO物体確認」から元の推論bbox・maskを物体ごとに確認する。担当者は画面上部へ入力する。未判定で器具対応済みの物体は「DINOに使う」を初期選択し、既存の不採用は維持する。「保存して次へ」で画像内の全物体の採否と画像承認を一括保存する。器具対応未確定の物体は採用できず、mask補正機能はない。YOLO用の矩形追加・削除・再ラベルは「YOLO注釈」で行う。

有効な画像承認と物体の採用判定に加え、正式maskの保存が揃った新規結果だけを、`sam3_dino_results.is_dino_ready`で公開可能とする。既存の`sam3_dino_objects`ビューも各セッションDB内で維持する。別の原本PNGとの二重保存は行わない。

[合意済みのDINO mask保存バグ修正仕様](docs/dino_mask_save_fix.md)（実装未実施）では、「保存して次へ」で今回採用したmaskだけをPNG保存し、その採用先に既存ファイルがある場合だけYesで上書き同意を求める。Noでは保存せず同じ画像に留まる。旧採用maskやバグ由来の不要maskは対象画像の再保存成功時に整理する。他画像・他セッション・推論原本・監査履歴・YOLO状態は変更せず、一括移動・削除は行わない。通常の保存失敗・SQL確定失敗では削除分も復元する。所有識別などの実装前確認事項、[撮影データ連携手順](docs/capture_workflow.md#承認用maskと上書き確認)、[公開契約](docs/contracts/dino_mask_publication.md)を参照する。

DINOでも器具タグ（UID＋個体番号）ごとに候補を手動採用できる。タグを選び、画像上のbboxまたは候補一覧からmaskを確認し、「選択タグにこのDINO候補を採用」を押す。低スコアや自動対応が曖昧な候補も選択可能である。同じタグの旧採用候補は確認後に解除し、1タグにつき最大1候補を採用する。「未判定候補を使わないにする」で余剰候補の採否を確定できる。全候補のmask確認後に保存する。新しい画面のDINO保存は未解決のYOLO注釈を変更・承認せず、有効なYOLO承認がなければDINO専用承認とする。既存の有効なYOLO承認・公開は維持する。レビューschema version 5で手動対応履歴を追加し、対応・採否・maskを同時に確定する。未承認のYOLO公開には別途YOLOの確認・承認が必要である。

採否と履歴、一覧確認状態、レビュー注釈・画像承認のスナップショットはSQLへ保持する。既にSQLへ記録済みの取消や注釈変更を、再起動時に残存JSONの承認で復活させない。旧レビュー用ファイルはYOLO互換と移行のため残す。DINOv3・YOLO側の読み取り変更は本プロジェクトの実装範囲に含めない。
