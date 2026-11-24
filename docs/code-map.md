# コードマップ

## 実行・処理

| パス | 主な責務 |
|---|---|
| [src/analyze.py](../src/analyze.py) | 撮影タグの器具単位でbbox・mask成功率と未採用bbox平均数を集計し、master DBの日本語器具名を使ってCSV・画像・失敗例を出力する |
| [src/capture_workflow.py](../src/capture_workflow.py) | CLI設定を読み込み、画像選択・推論・レビューを選択server再利用で繰り返す |
| [src/sam3_pipeline/usecase/capture.py](../src/sam3_pipeline/usecase/capture.py) | 撮影サンプルを推論結果とレビュー草案へつなぐ |
| [src/sam3_pipeline/infrastructure/capture_predictor.py](../src/sam3_pipeline/infrastructure/capture_predictor.py) | SAM3共通プロンプト推論を行い、設定有効時はYOLOの全bboxをvisual promptとして追加し、SAM3候補だけを返す |
| [src/yolov12/inference.py](../src/yolov12/inference.py) | settings_yolo.yamlから重みを解決し、単体用既定値またはcapture指定confidenceでYOLO検出する |
| [src/sam3_pipeline/infrastructure/capture_source.py](../src/sam3_pipeline/infrastructure/capture_source.py) | targetsの範囲にある安全なセッションDBから検証済みの撮影画像・タグを読み取る |
| [src/sam3_pipeline/infrastructure/selection_server.py](../src/sam3_pipeline/infrastructure/selection_server.py) | 画像内容の読み込み前に対象を選び、同じserverを維持してレビュー後の再選択を受け付ける |
| [src/sam3_pipeline/infrastructure/review_server.py](../src/sam3_pipeline/infrastructure/review_server.py) | ローカルレビュー画面とレビューAPIを提供し、run時は既存選択画面への復帰をCLIへ通知する。YOLO一括公開時の注釈検証を画像・shape／fieldが特定できる文脈付きで報告し、DINO一覧に永続化状態を反映する |
| [src/sam3_pipeline/infrastructure/annotation_store.py](../src/sam3_pipeline/infrastructure/annotation_store.py) | YOLO注釈・画像承認を保存し、公開用rectangle・UID検証を行う。未編集の初期旧注釈だけをタグレビュー形式へ移行し、公開時の検証エラーでshape／fieldを特定できる情報を返す |
| [src/sam3_pipeline/infrastructure/object_review_store.py](../src/sam3_pipeline/infrastructure/object_review_store.py) | DINO候補のタグ別採否、承認用mask、公開状態をセッションDBへ保存・読出しする。DINO一覧状態の正本は永続化された判定／承認であり、mask存在情報は判定根拠にしない |
| [src/sam3_pipeline/interface/selection.html](../src/sam3_pipeline/interface/selection.html) | セッション・複数画像を選び、状態を確認するUIを提供する |
| [src/sam3_pipeline/interface/tag_review.js](../src/sam3_pipeline/interface/tag_review.js) | YOLO矩形とタグの対応編集、およびDINO候補のタグ単位採用・同タグ候補の不採用化を管理する |
| [src/sam3_pipeline/interface/review.html](../src/sam3_pipeline/interface/review.html) | ブラウザー上の画像レビューUI、選択セッション単位のYOLO一括確認、DINO部分判定の保存、明示的な作業完了と既存画像選択画面への復帰操作を提供する |

## 設計・運用文書

| パス | 内容 |
|---|---|
| [docs/capture_workflow.md](capture_workflow.md) | 撮影から推論・レビュー・公開までの手順 |
| [src/configs/analyze.yaml](../src/configs/analyze.yaml) | 解析対象条件と出力先`output_root`の実行時設定 |
| [docs/analysis_specification.md](analysis_specification.md) | 全体および器具種別ごとの解析指標・対象範囲・出力仕様 |
| [docs/sam3_yolo_bbox_inference.md](sam3_yolo_bbox_inference.md) | capture設定によるYOLO bboxのSAM3 visual prompt入力仕様 |
| [docs/review_ui_improvements.md](review_ui_improvements.md) | 合意済みのYOLO/DINOレビューUI改善 |
| [docs/selective_inference_and_review.md](selective_inference_and_review.md) | 推論前の画像選択と選択画像だけのレビュー仕様 |
