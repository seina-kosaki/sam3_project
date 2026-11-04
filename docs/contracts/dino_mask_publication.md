# DINO正式maskの保存・公開契約

状態: [修正仕様](../dino_mask_save_fix.md)に対応する実装済み契約。手動確認待ち。HTTP API・DBスキーマの追加変更はない。

追加合意（未実装）: [レビューUI改善仕様](../review_ui_improvements.md)の「すべて不採用で保存して次へ」では、UIから全候補を`rejected`とする既存の`POST /api/complete-review`入力を送り、既存採用も解除する。この操作では個別選択・閲覧を必須としない。採用PNG新規保存なし、対象画像の不要mask整理、全不採用時の公開対象0件、失敗時維持／復元は以下の契約に従う。YOLO確認状態・承認・公開状態は変更せず、保存成功後のみ次画像へ進む。新しいHTTP項目やDB列は本合意では決定しない。

## 提供側・利用側の入出力

| 提供側 → 利用側 | 入力 | 出力・意味 |
|---|---|---|
| レビューサーバー → UI | `GET /api/items` | 画像の`revision`と候補の`object_index`、`dino_status`、`saved_mask_exists`等。存在情報は取得時点の正式保存先の実ファイル有無であり、採用・承認の証拠ではない |
| UI → レビューサーバー | `POST /api/complete-review` | `revision: string`、`reviewer: string`、`confirm_complete: true`、全候補の`decisions: [{object_index: integer, status: "accepted" または "rejected"}]`。既存の`assignments`、`overwrite_masks`、`allow_empty`、`exception_note`も引き続き使用する |
| SAM3保存処理 → DINO取得側 | 現行画像・推論・確定採否・対応・画像承認 | 所属セッションDBの`sam3_dino_results`、`sam3_dino_objects`および採用mask PNG。取得側は公開条件と実ファイルの存在・hashを照合する |

`assignments`は既存どおり省略可能。指定時は全候補の`object_index`、`instrument_uid: string または null`、`instance_no: integer または null`を渡し、未対応は両方nullにする。`overwrite_masks`は省略時は同意なしで、trueだけが明示同意。`allow_empty`／`exception_note`と、`assignments`省略時の従来の画像承認条件は変更しない。

UIは未保存の今回採否で採用対象を選び、最新存在情報を同じレビュー版・候補indexで照合する。サーバーも今回採用する同じ保存先だけに上書き同意を適用する。不採用ファイルの存在・削除は上書き確認の条件ではない。

## SQLiteの関連項目

各撮影セッションの`metadata.sqlite3`を正本とする。以下は本修正に関連する既存列の抜粋であり、列追加・型変更の定義ではない。必須性は既存のNOT NULL／主キー定義に従う。

| テーブル | カラム | SQLite型 | 必須 | 意味 |
|---|---|---|---|---|
| `sam3_review_revisions` | `revision` | TEXT | 主キー | レビュー版 |
| 同上 | `session_id`, `image_id`, `prediction_id` | TEXT | 各必須 | 所属画像・推論の識別 |
| 同上 | `is_current`, `status` | INTEGER, TEXT | 各必須 | 現行版・承認状態 |
| 同上 | `approval_id`, `approval_json` | TEXT | 任意 | 承認参照・記録 |
| `sam3_object_reviews` | `revision`, `object_index` | TEXT, INTEGER | 複合主キー・必須 | 候補の固定識別 |
| 同上 | `instrument_uid`, `instance_no` | TEXT, INTEGER | 任意 | 器具UID＋個体番号 |
| 同上 | `dino_status` | TEXT | 必須 | pending／accepted／rejected |
| `sam3_approved_masks` | `revision`, `object_index` | TEXT, INTEGER | 複合主キー・必須 | 承認用maskの候補識別 |
| 同上 | `mask_path`, `mask_sha256` | TEXT | 各必須 | 正式PNGの共有ルート相対パス・hash |
| `sam3_dino_results` | `revision`, `object_index` | TEXT, INTEGER | 複合主キー・必須 | 候補単位の公開スナップショット |
| 同上 | `session_id`, `image_id`, `prediction_id` | TEXT | 各必須 | 所属画像・推論 |
| 同上 | `mask_path`, `mask_sha256` | TEXT | 各必須 | 正式mask参照、または既存の推論参照（`memory:`を含む） |
| 同上 | `review_status`, `is_dino_ready` | TEXT, INTEGER | 各必須 | 採否と公開可能フラグ |

`sam3_dino_objects`は既存の公開ビュー。`sam3_object_review_events`、`sam3_object_assignment_events`、`sam3_review_events`の既存監査履歴および推論原本テーブルは削除しない。

## 保存・読み取り可能時点

- 確認用maskの保持・推論情報の記録だけでは正式PNG保存・DINO公開としない。
- 対象画像の保存成功時に、今回acceptedのPNGだけを作成／置換し、対象画像が所有する不要な承認用maskを整理する。採用切替前のmaskと既存バグ由来の不採用maskを含む。他画像・他セッションへの一括整理はしない。
- 採否・対応・承認・承認用mask台帳・DINO公開スナップショットを同じ保存処理で確定する。通常の保存失敗とSQL確定失敗では、削除分を含むファイルとDB状態を維持／復元する。プロセス強制終了への保証は従来どおり追加しない。
- 下流は確定後の`is_dino_ready=1`または公開ビューを使う。現行版、有効な画像承認、accepted、器具対応、撮影タグ版・原本情報の一致など既存条件を維持する。全候補不採用なら公開対象は0件。
- 保存時、対象画像の全レビュー版の`sam3_approved_masks`行を外し、今回acceptedの現行版だけに正式PNGのパス・hashを登録する。同じトランザクションの公開同期により、旧版／不採用候補の`sam3_dino_results`は推論参照（`memory:`を含む）へ戻り、旧版は非公開、不採用は`is_dino_ready=0`となる。不採用の判定行と採否・対応・レビューイベントは保持する。履歴中の当時のパス・hashは現存保証ではない。
- PNGの実ファイル検証は取得側でも行う。DBとファイルシステムをまたぐクラッシュ原子性は保証しない。
- 所有は所属DBの承認用mask台帳とレビュー版の`session_id`＋`image_id`から確定する。他画像と共有するパス、推論／撮影原本と共有するパス、セッションの`masks`外の台帳を変更しない。現行／過去候補の完全一致の保存先に台帳未登録ファイルがある場合は保存を停止し、その他の所有不明ファイルは整理対象にしない。
- DINO操作でYOLO矩形・承認・公開状態、推論原本を変更しない。
