# YOLO確認・部分公開契約

状態: 既存の[UI改善仕様](../review_ui_improvements.md)に対応する公開契約は実装済み（2026-11-04、手動確認待ち）。追加公開は下記「実装した公開方式」の重複しない差分バッチとした。2026-11-13に合意した`POST /api/reviewed/batch`は実装済み・自動テスト済みで、手動確認待ち。公開済み画像の変更版の再公開は未決のため保留扱いとする。

## 提供側・利用側の入出力

| 提供側 → 利用側 | 入力 | 出力・意味 |
|---|---|---|
| UI → レビューサーバー | `POST /api/reviewed`: `revision: string`、`reviewed: boolean`。既存の`allow_empty`、`exception_note`も使用 | 画像単位の現行YOLO注釈の確認状態。従来どおり維持し、DINO採否・承認とは独立 |
| UI → レビューサーバー | `POST /api/reviewed/batch`: `session_id: string` | 指定セッションの現行・最新レビュー画像から、未確認かつ保存済み採用bboxがある画像をサーバー側で判定してYOLO確認状態を一括保存 |
| UI → レビューサーバー | `POST /api/publish`: 現行は`session_id: string`、`reviewer: string`、`confirm_complete: true` | 現行はセッション全画像確認後、`exports/<session_id>/manifest.json`を新規公開。変更後はYOLO確認済みの今回対象だけを公開し、未確認を除外。全画像確認を意味する現行`confirm_complete`の扱いは実装前に整理する |
| レビューデータ → UI | `GET /api/items`の`annotation` | 新規草案は`tag_review_version: 1`を使用し、一意に対応したshapeだけを`shapes`へ、未解決shapeはラベルを付けず`review_candidates`へ分ける。旧形式も読み込み可能で、初期生成時と完全一致する未確認・未承認の草案だけを自動移行する |
| SAM3公開処理 → YOLO取得側 | 現行承認・注釈・撮影原本参照 | schema version 1のmanifestとLabelme注釈、および所属セッションDBの`sam3_yolo_results`。画像原本は複製しない |

公開前のUI表示は「公開対象○件／未確認○件」。今回対象に同一データの公開済み分を再登録しない。対象0件では公開しない。bbox等のYOLO編集後は確認を解除し、再確認・必要な再承認を経る。

## YOLO確認API

### 画像単位確認（既存）

- `POST /api/reviewed`は既存契約を維持する。`revision`と`reviewed`、既存の`allow_empty`／`exception_note`を受け取り、画像単位でYOLO確認状態を保存する。
- 既存クライアントとの互換性を保ち、このAPIを一括確認APIに置き換えない。

### セッション単位一括確認（追加）

- **HTTP**: `POST`
- **Path**: `/api/reviewed/batch`
- **Request body**: `{"session_id": "<session_id>"}`
- **対象の決定**: 指定セッションに属する現行・最新レビュー画像をサーバー側で調べ、未確認であり、保存済み注釈に採用bboxが1件以上ある画像だけを対象にする。レビュー版の現行性もサーバー側で判定する。
- **対象外**: bboxがない画像。候補の採否が未確定／不採用で採用bboxがない画像も対象外。信頼度、ページネーション、UI表示・フィルターには依存しない。
- **更新内容**: 対象画像のYOLO確認状態だけを確認済みにする。注釈、候補の採否、DINO状態、承認、公開状態を変更しない。既存の`allow_empty`と`exception_note`は変更せず保持する。
- **成功Response (HTTP 200)**: `{"confirmed": <成功件数>, "failed": [{"revision": "<revision>", "error": "<error>"}]}`。`failed`は画像ごとの保存失敗であり、部分失敗では成功分だけを反映し、失敗した画像は未確認のままとする。対象画像が0件の場合もHTTP 200で`{"confirmed": 0, "failed": []}`を返し、保存処理は行わない。
- **全体要求エラー (HTTP 400)**: `session_id`が文字列でない、または有効なセッションID形式でない場合は、既存のJSONエラー形式`{"error": "<message>"}`を返す。
- 未処理画像を成功扱いにしない。既存の`POST /api/reviewed`の応答・動作は変更しない。
- **互換性**: 既存`POST /api/reviewed`のメソッド・path・入力・画像単位動作は維持する。新API追加によって既存APIを廃止・変更しない。

## SQLiteの関連項目

所属セッションの`metadata.sqlite3`が正本。下表は公開に関係する既存列の抜粋で、列追加・型変更は本仕様では決定しない。

| テーブル | カラム | SQLite型 | 必須 | 意味 |
|---|---|---|---|---|
| `annotation_revisions` | `session_id`, `image_id`, `revision` | TEXT | 複合主キー・必須 | 公開台帳の画像・版識別 |
| 同上 | `tag_revision` | INTEGER | 必須 | 撮影タグ版 |
| 同上 | `status`, `annotation_path`, `annotation_sha256`, `updated_at` | TEXT | 必須 | 状態・注釈参照・更新日時 |
| 同上 | `approval_id`, `reviewer`, `approved_at`, `manifest_path`, `exported_at` | TEXT | 任意 | 承認と公開完了の参照 |
| `sam3_yolo_results` | `revision` | TEXT | 主キー | レビュー版 |
| 同上 | `session_id`, `image_id`, `source_path`, `source_sha256` | TEXT | 必須 | 所属画像・原本参照 |
| 同上 | `tag_revision` | INTEGER | 必須 | タグ版 |
| 同上 | `annotation_json`, `source_json`, `status`, `exception_note` | TEXT | 必須 | 注釈・原本スナップショットと状態 |
| 同上 | `allow_empty`, `is_yolo_ready` | INTEGER | 必須 | 空画像確認・公開可能フラグ |
| 同上 | `annotation_sha256`, `annotation_path`, `approval_json`, `approval_id`, `reviewer`, `approved_at`, `manifest_path`, `exported_at` | TEXT | 任意 | 注釈・承認・公開参照 |

## 保存・読み取り可能時点

- YOLO確認だけでは公開しない。現行の承認・注釈hashに一致したmanifestの公開完了後、原本・撮影タグ版など既存条件を満たす`is_yolo_ready=1`を下流の読取条件とする。取得側もファイル存在・hashを検証する。
- 変更後はセッションの未確認画像の有無で確認済み画像の公開を妨げない。承認とexportの両方を今回対象に合わせ、除外画像を公開完了として記録しない。
- 部分公開で残した画像は後日追加公開でき、同一の公開済みデータは重複登録しない。DINO採否・承認によってYOLO確認・公開を成立させない。
- 現行のexportは既存出力先を上書きせず、同一画像の複数レビュー版の同時出力を拒否する。注釈変更後は旧承認をそのまま利用できない。撮影タグ更新などによる失効条件も維持する。
- 追加公開のmanifest配置・累積／差分方式、同一データ照合の詳細、公開済み変更版の承認／旧出力との関係は未決。既存出力の自動削除・上書き・自動承認取消を本合意で許可したものとはしない。
- 現行の下流手順は「累積バッチなら最新manifestだけ」「複数manifestは相互に重複しない場合だけ」。この前提を壊す配置や運用を未合意で導入しない。詳細確定後、本契約を実装と同じ変更内で更新する。

## 実装した公開方式（2026-11-04）

既存公開物を破壊せず、同一データを重複登録しない最も単純な方式として次を採用した。DB列・manifest schema versionは変更していない。

- `GET /api/items`の各画像に`published: boolean`（同一データ公開済み）と`published_other_version: boolean`（同じ画像の別版・別注釈hashが公開済み）を追加した。既存manifest（`exports/*/manifest.json`）の`source.session_id`・`source.image_id`・`revision`・`labelme_sha256`の4項目一致を「同一データ」とする。画像IDだけでは同一とみなさない。
- `POST /api/publish`の`confirm_complete: true`は「YOLO確認済み画像だけを公開する」ことの明示確認に意味を変更した（全画像確認は要求しない）。入力形式は変更なし。
- 公開対象 = 選択セッションでYOLO確認済み、かつ`published`でも`published_other_version`でもない画像。未確認は除外。対象0件は400エラーで拒否し、承認・exportを行わない。
- 承認とexportの両方を公開対象のレビュー版だけに限定する（`export_approved(..., revisions=...)`）。除外画像を公開完了として台帳に記録しない。
- 公開先: 初回は従来どおり`exports/<session_id>/manifest.json`。既に存在する場合は`exports/<session_id>+NNN/manifest.json`（NNN=002からの連番）へ新規作成する。`+`はセッションIDに使えない文字のため他セッションの公開先と衝突しない。既存出力は上書き・削除しない。
- 各manifestは今回の差分だけを含み、同一画像は他manifestと重複しない。現行下流手順の「複数manifestは相互に重複しない場合だけ」に適合する（「累積バッチなら最新manifestだけ」の運用では追加分だけしか読まれないため、下流は`exports/<session_id>`と`exports/<session_id>+*`をすべて読む必要がある）。`sync_existing_exports`は`exports/*/manifest.json`を走査するため追加バッチも台帳同期対象となる。
- 公開済み画像の変更版（`published_other_version`）は、重複・置換を伴うため自動公開せず保留件数として表示する。旧承認の取消・旧manifestの扱いは未決のまま。
- UIの承認表示（ファイル承認）は注釈hashが現行注釈と一致する場合だけ承認済みとし、編集後に旧承認だけでYOLO確認済みに戻さない。
