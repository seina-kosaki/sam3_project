# YOLO確認・部分公開契約

状態: [UI改善仕様](../review_ui_improvements.md)に対応する合意済み変更予定。未実装。追加公開の出力構成と公開済み変更版は未決であり、以下では現行契約と確定変更を区別する。

## 提供側・利用側の入出力

| 提供側 → 利用側 | 入力 | 出力・意味 |
|---|---|---|
| UI → レビューサーバー | `POST /api/reviewed`: `revision: string`、`reviewed: boolean`。既存の`allow_empty`、`exception_note`も使用 | 現行YOLO注釈の確認状態。DINO採否・承認とは独立 |
| UI → レビューサーバー | `POST /api/publish`: 現行は`session_id: string`、`reviewer: string`、`confirm_complete: true` | 現行はセッション全画像確認後、`exports/<session_id>/manifest.json`を新規公開。変更後はYOLO確認済みの今回対象だけを公開し、未確認を除外。全画像確認を意味する現行`confirm_complete`の扱いは実装前に整理する |
| SAM3公開処理 → YOLO取得側 | 現行承認・注釈・撮影原本参照 | schema version 1のmanifestとLabelme注釈、および所属セッションDBの`sam3_yolo_results`。画像原本は複製しない |

公開前のUI表示は「公開対象○件／未確認○件」。今回対象に同一データの公開済み分を再登録しない。対象0件では公開しない。bbox等のYOLO編集後は確認を解除し、再確認・必要な再承認を経る。

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
