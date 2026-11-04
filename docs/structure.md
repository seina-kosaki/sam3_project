# セッション別SAM3出力の構成

## 配置の理由

DB解決と公開スナップショットの更新を既存の各保存ストアで共有し、セッション境界と公開条件の重複実装を防ぐため、infrastructure層へ二つの責務を追加した。

| ファイル | 責務 |
|---|---|
| [capture_workflow.py](../src/capture_workflow.py) | 設定した共有ルートを各保存ストアへ注入し、既存の実行段階を接続する |
| [session_database.py](../src/sam3_pipeline/infrastructure/session_database.py) | 既存DBを探索し、セッションID・レビュー版から所有DBを特定する。DBを新規作成しない |
| [prediction_store.py](../src/sam3_pipeline/infrastructure/prediction_store.py) | 推論情報をセッションDBへ記録し、maskをメモリー上で保持する。必要時は遅延再推論する |
| [object_review_store.py](../src/sam3_pipeline/infrastructure/object_review_store.py) | 所属セッションへ画像承認・物体採否・レビュー履歴を保存する |
| [annotation_ledger.py](../src/sam3_pipeline/infrastructure/annotation_ledger.py) | 所属セッションの承認台帳へ従来のYOLO互換export参照を記録する |
| [handoff_store.py](../src/sam3_pipeline/infrastructure/handoff_store.py) | 同じトランザクション内でDINO・YOLO専用テーブルと公開フラグを同期する |
| [approved_masks.py](../src/sam3_pipeline/infrastructure/approved_masks.py) | 承認用maskのセッション内配置・上書き同意・保存失敗時のファイル復元を担当する |
| [tag_annotations.py](../src/sam3_pipeline/infrastructure/tag_annotations.py) | YOLO採用矩形と可視タグの一対一対応を保存・承認時に検証する |
| [tag_review.js](../src/sam3_pipeline/interface/tag_review.js) | DOMから独立したタグ・bbox選択、対応付け、手描きと保存形式の変換を担当する |

承認用maskの可変ファイル操作を推論原本の不変保存から分離するため、専用ファイルを追加した。

タグ対応の編集規則をHTMLのDOM操作から分離し、保存側の対応検証も共有することで、画面の肥大化と保存・承認間の検証漏れを抑える。

DINOのタグ別候補選択も同じ編集スクリプトへ置くが、YOLO注釈とは別の状態として保持する。手動対応は既存の物体レビュー保存先で採否・maskと同時に確定し、schema version 5の対応変更履歴へ記録する。DINO専用承認とYOLO承認の取得条件を区別し、未解決のYOLO候補がDINO確認を妨げないようにする。

## 保存先

```text
shared_data/
  YYYY-MM-DD/<session_id>/
    metadata.sqlite3
    raw/
    depth/
    masks/
      <画像名>_<物体番号>.png

annotation_work/
  drafts/
  reviews/
  approvals/
  latest/
  exports/<session_id>/
    manifest.json
    items/<識別子>/annotation.json
```

各セッションDBが撮影とSAM3結果の正本である。DINO用の推論原本とYOLO用の編集済み注釈を分離し、公開テーブルは既存の記録からのみ更新する。旧単一DBは使わない。画像とmaskのパスは共有データルート基準、注釈の作業パスは設定した作業ルート基準である。

承認前のmaskはメモリー上の確認用データであり、PNGは保存しない。DINOの「保存して次へ」で、各物体のmaskをセッション直下の`masks`へ初めてPNG保存する。保存済みならYesで明示同意した場合だけ置換する。`sam3_approved_masks`に相対パスとhashを記録し、DINO結果テーブルと公開ビューへ反映する。再起動時は保存済みmaskを検証して読むか、確認時に同じ設定でSAM3を再実行する。旧`annotations/sam3`等の一括移動・削除は行わない。

## 境界

- domain／usecaseのポートは変更せず、ストア内部でセッションへ振り分ける。
- ストアへ明示的なDBパスを注入する場合は単一セッションの内部利用である。通常CLIは常に共有ルートを渡す。
- セッションDB・レビュー版の欠損や重複はエラーとし、他のDBへ退避しない。
- DINOv3・YOLO側の読み取り変更、学習全体の台帳、日次レポートは今回の対象外である。
