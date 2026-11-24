from contextlib import closing
from pathlib import Path
import sqlite3


CAPTURE_DAY_DIR = Path(r"C:\work\SPD\shared_data\2026-11-24")

QUERY = """
SELECT
    i.session_id,
    i.image_id,
    h.instrument_uid,
    COUNT(*) AS tag_rows,
    COUNT(DISTINCT h.instance_no) AS distinct_instances
FROM images AS i
JOIN hints AS h ON h.image_id = i.image_id
WHERE i.tag_status = 'tagged'
  AND h.visible = 1
GROUP BY i.session_id, i.image_id, h.instrument_uid
ORDER BY i.session_id, h.instrument_uid, i.image_id
"""


if not CAPTURE_DAY_DIR.is_dir():
    raise FileNotFoundError(f"撮影データの日付フォルダーがありません: {CAPTURE_DAY_DIR}")

databases = sorted(CAPTURE_DAY_DIR.glob("*/metadata.sqlite3"))
if not databases:
    raise FileNotFoundError(f"セッションDBが見つかりません: {CAPTURE_DAY_DIR}")

by_session: dict[tuple[str, str], dict[str, int]] = {}
overall: dict[str, dict[str, int]] = {}
multiple_tag_images: list[tuple[str, str, str, int, int]] = []

for database in databases:
    # mode=ro でDBを読み取り専用にする。
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    ) as connection:
        rows = connection.execute(QUERY).fetchall()

    for session_id, image_id, instrument_uid, tag_rows, distinct_instances in rows:
        session_key = (session_id, instrument_uid)

        if session_key not in by_session:
            by_session[session_key] = {
                "image_uid_count": 0,
                "distinct_tag_count": 0,
                "tag_row_count": 0,
            }

        if instrument_uid not in overall:
            overall[instrument_uid] = {
                "image_uid_count": 0,
                "distinct_tag_count": 0,
                "tag_row_count": 0,
            }

        for counts in (by_session[session_key], overall[instrument_uid]):
            counts["image_uid_count"] += 1
            counts["distinct_tag_count"] += distinct_instances
            counts["tag_row_count"] += tag_rows

        if distinct_instances > 1 or tag_rows > 1:
            multiple_tag_images.append(
                (session_id, image_id, instrument_uid, distinct_instances, tag_rows)
            )

print("セッション別（image_uid_count が画像ID×器具IDの件数）")
print("session_id\tinstrument_uid\timage_uid_count\tdistinct_tag_count\ttag_row_count")
for (session_id, instrument_uid), counts in sorted(by_session.items()):
    print(
        f"{session_id}\t{instrument_uid}\t"
        f"{counts['image_uid_count']}\t"
        f"{counts['distinct_tag_count']}\t"
        f"{counts['tag_row_count']}"
    )

print("\n全セッション合計")
print("instrument_uid\timage_uid_count\tdistinct_tag_count\ttag_row_count")
for instrument_uid, counts in sorted(overall.items()):
    print(
        f"{instrument_uid}\t"
        f"{counts['image_uid_count']}\t"
        f"{counts['distinct_tag_count']}\t"
        f"{counts['tag_row_count']}"
    )

print("\n同じ画像・器具IDに複数のタグ行がある箇所")
print("session_id\timage_id\tinstrument_uid\tdistinct_instances\ttag_rows")
for row in multiple_tag_images:
    print("\t".join(str(value) for value in row))