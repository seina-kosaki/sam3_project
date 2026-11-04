/**
 * タグとYOLO矩形の対応・選択を管理する。
 * 画面のDOM操作から対応規則を分離し、撮影タグの個体単位で一対一を維持する。
 * 保存時は採用矩形だけをshapesへ、比較用候補はreview_candidatesへ分ける。
 * 関連: review.html、infrastructure/tag_annotations.py。DINO原本は変更しない。
 */
"use strict";

const TagReview = (() => {
  const VERSION = 1;
  const UNRESOLVED = "__unresolved__";
  const COLORS = ["#d32f2f", "#1565c0", "#8e24aa", "#008577", "#e36c09", "#5d4037", "#c2185b", "#455a64"];
  const key = tag => JSON.stringify([tag.instrument_uid, tag.instance_no]);
  const color = index => COLORS[index] || `hsl(${(index * 137.508) % 360} 65% 38%)`;
  const boxName = (shape, index) => Number.isInteger(shape.group_id)
    ? `Box ${String(shape.group_id).padStart(3, "0")}` : `手描き ${String(shape.manual_box_id ?? index + 1).padStart(3, "0")}`;

  /** 候補を学習用shapesから外しても、保存済み推論スコアを表示する。 */
  function score(shape) {
    try {
      const value = JSON.parse(shape.description || "{}").score;
      return typeof value === "number" && Number.isFinite(value) ? value : null;
    } catch { return null; } // 旧形式の自由記述を数値として推測しない。
  }

  /** 旧注釈は個体を一意に判断できる場合だけ対応として扱う。 */
  function identity(shape, tags) {
    if (shape.tag_assignment) return tags.find(tag => key(tag) === key(shape.tag_assignment) && shape.label === tag.instrument_uid) || null;
    if (shape.label === UNRESOLVED) return null;
    let instance = null;
    if (shape.description) {
      try { instance = JSON.parse(shape.description).instance_no ?? null; }
      catch { return null; } // 不正な旧情報から個体を推測せず、画面で再対応を求める。
    }
    const matches = tags.filter(tag => tag.instrument_uid === shape.label && (instance === null || tag.instance_no === instance));
    return matches.length === 1 ? matches[0] : null;
  }

  class Editor {
    /** 編集用複製を持ち、画面を開いただけでは保存済み注釈を変更しない。 */
    constructor(item) {
      this.tags = item.tags.filter(tag => tag.visible);
      this.entries = [...item.annotation.shapes, ...(item.annotation.review_candidates || [])]
        .map(shape => ({shape: structuredClone(shape), tag: identity(shape, this.tags)}));
      // 配列の並びが保存時に変わっても、手描き矩形の表示IDを維持する。
      this.nextManualId = Math.max(0, ...this.entries.map(entry => entry.shape.manual_box_id || 0)) + 1;
      for (const {shape} of this.entries) {
        if (shape.group_id === null && !shape.manual_box_id) shape.manual_box_id = this.nextManualId++;
      }
      this.selectedTag = this.tags.length ? 0 : null;
      this.selectedBox = null;
      this.mode = "confirm";
    }

    /** 対象タグに採用されている矩形の番号を返す。 */
    assigned(tagIndex) {
      return this.entries.flatMap((entry, index) => entry.tag && key(entry.tag) === key(this.tags[tagIndex]) ? [index] : []);
    }

    /** 同種器具の個体番号と表示番号を併記する。 */
    tagName(index) {
      const tag = this.tags[index];
      return `タグ ${index + 1} · ${tag.instrument_name || tag.instrument_uid} · ${tag.instrument_uid} / 個体 ${tag.instance_no}`;
    }

    /** 画像上の重なった矩形を全て返し、選択を巡回できるようにする。 */
    hitBoxes(point) {
      return this.entries.flatMap(({shape}, index) => {
        const [[x1, y1], [x2, y2]] = shape.points;
        return point.x >= Math.min(x1, x2) && point.x <= Math.max(x1, x2)
          && point.y >= Math.min(y1, y2) && point.y <= Math.max(y1, y2) ? [index] : [];
      });
    }

    /** bboxを選ぶと対応先タグも選択し、未対応の場合は対象タグを維持する。 */
    selectBox(index) {
      this.selectedBox = index;
      const tag = this.entries[index].tag;
      if (tag) this.selectedTag = this.tags.findIndex(item => key(item) === key(tag));
    }

    /** 別タグの候補または既存対応を置換する前に確認を必要とする。 */
    needsReplacement(index) {
      const tag = this.tags[this.selectedTag];
      if (!tag) throw new Error("先に器具タグを選択してください");
      return this.assigned(this.selectedTag).some(value => value !== index)
        || Boolean(this.entries[index]?.tag && key(this.entries[index].tag) !== key(tag));
    }

    /** 明示された対象タグへ一つだけ対応付け、古い対応は候補へ戻す。 */
    adopt(index) {
      const tag = this.tags[this.selectedTag];
      if (!tag || !this.entries[index]) throw new Error("タグとbboxを選択してください");
      for (const previous of this.assigned(this.selectedTag)) this.entries[previous].tag = null;
      this.entries[index].tag = tag;
      this.selectedBox = index;
    }

    /** 手描きは推論候補番号を付けず、YOLO専用の新しい矩形として扱う。 */
    draw(points) {
      if (this.selectedTag === null) throw new Error("先に器具タグを選択してください");
      const index = this.entries.length;
      this.entries.push({shape: {label: UNRESOLVED, points, group_id: null, manual_box_id: this.nextManualId++, shape_type: "rectangle",
        flags: {}, description: JSON.stringify({manual: true})}, tag: null});
      this.adopt(index);
      this.mode = "confirm";
    }

    /** 採用矩形と比較用候補を分け、個体番号を明示して保存する。 */
    annotation(original) {
      const shapes = [], candidates = [];
      for (const {shape, tag} of this.entries) {
        const copy = structuredClone(shape);
        copy.label = tag ? tag.instrument_uid : UNRESOLVED;
        copy.tag_assignment = tag ? {instrument_uid: tag.instrument_uid, instance_no: tag.instance_no} : null;
        (tag ? shapes : candidates).push(copy);
      }
      return {...original, tag_review_version: VERSION, shapes, review_candidates: candidates};
    }
  }
  class DinoEditor {
    /** 推論原本に触れず、画面上のDINO候補の対応と採否だけを編集する。 */
    constructor(item) {
      this.tags = item.tags.filter(tag => tag.visible);
      this.objects = item.objects || [];
      this.selectedTag = this.tags.length ? 0 : null;
    }

    /** 同種器具も個体番号で区別した採用候補を返す。 */
    assigned(index) {
      const tag = this.tags[index];
      return this.objects.filter(object => tag && object.dino_status === "accepted"
        && object.instrument_uid === tag.instrument_uid && object.instance_no === tag.instance_no);
    }

    /** bbox選択時に対応するタグを選択し、未対応なら対象タグを維持する。 */
    select(object) {
      const index = this.tags.findIndex(tag => tag.instrument_uid === object.instrument_uid && tag.instance_no === object.instance_no);
      if (index >= 0) this.selectedTag = index;
    }

    /** 候補の付け替えで既存の採用を解除する必要があるか返す。 */
    needsReplacement(object) {
      const tag = this.tags[this.selectedTag];
      if (!tag) throw new Error("先に器具タグを選択してください");
      return this.assigned(this.selectedTag).some(value => value !== object)
        || Boolean(object.instrument_uid && (object.instrument_uid !== tag.instrument_uid || object.instance_no !== tag.instance_no));
    }

    /** 一つのタグに採用候補を一つだけ対応付け、旧候補は不採用へ戻す。 */
    adopt(object) {
      const tag = this.tags[this.selectedTag];
      if (!tag || !object) throw new Error("タグとDINO候補を選択してください");
      for (const previous of this.objects) {
        if (previous !== object && previous.instrument_uid === tag.instrument_uid && previous.instance_no === tag.instance_no) {
          previous.instrument_uid = null;
          previous.instance_no = null;
          previous.dino_status = "rejected";
        }
      }
      object.instrument_uid = tag.instrument_uid;
      object.instance_no = tag.instance_no;
      object.dino_status = "accepted";
    }
  }
  return {Editor, DinoEditor, color, identity, key, boxName, score};
})();
