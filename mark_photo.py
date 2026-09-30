# -*- coding: utf-8 -*-
"""
合照标注 + 人数统计工具
基于 GitHub 项目 ageitgey/face_recognition
https://github.com/ageitgey/face_recognition

目录约定（放在本脚本同级目录）：
  known/   单人照片库。两种布局二选一：
           (A) known/张三.jpg, known/李四.jpg ...  文件名即姓名
           (B) known/张三/photo1.jpg, known/李四/photo2.jpg ...  子文件夹名即姓名
  test/    待标注的合照（jpg/jpeg/png）
  output/  标注结果图输出目录

用法：
  .venv\\Scripts\\python.exe mark_photo.py
"""

import os
import sys
import pickle
import datetime
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
KNOWN_DIR = ROOT / "known"
TEST_DIR = ROOT / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "known_faces.pkl"
ROSTER_PATH = ROOT / "roster.txt"   # 花名册：每行一个姓名

# 允许的图片后缀
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 人脸识别阈值：越小越严格（默认 0.6）
TOLERANCE = 0.55

# Windows 中文字体
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_roster() -> set:
    """读取花名册 roster.txt，每行一个姓名。返回姓名集合（已去空白/BOM）。"""
    if not ROSTER_PATH.exists():
        return set()
    names = set()
    # 优先 UTF-8-sig（自动去 BOM），失败则按 GBK 读
    try:
        text = ROSTER_PATH.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = ROSTER_PATH.read_text(encoding="gbk", errors="ignore")
    for line in text.splitlines():
        name = line.strip()
        if name:
            names.add(name)
    return names


def iter_known_images():
    """
    产出 (姓名, 图片路径) 列表。
    支持两种布局：
      A: known/张三.jpg
      B: known/张三/xxx.jpg
    """
    results = []
    if not KNOWN_DIR.exists():
        return results

    for entry in sorted(KNOWN_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMG_EXT:
            # 布局 A：文件名即姓名
            name = entry.stem
            results.append((name, entry))
        elif entry.is_dir():
            # 布局 B：子文件夹即姓名
            name = entry.name
            for img in sorted(entry.iterdir()):
                if img.is_file() and img.suffix.lower() in IMG_EXT:
                    results.append((name, img))
    return results


def build_known_faces(force: bool = False):
    """注册单人照片库，保存为 pkl。"""
    if MODEL_PATH.exists() and not force:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    records = iter_known_images()
    if not records:
        print("[错误] known/ 目录下没有找到单人照片。")
        print("       请把单人照片放到 known/ 下，文件名（或子文件夹名）即姓名。")
        sys.exit(1)

    known_encodings = []
    known_names = []
    skipped = 0
    for name, img_path in records:
        try:
            image = face_recognition.load_image_file(str(img_path))
            locs = face_recognition.face_locations(image)
            if len(locs) == 0:
                print(f"  [跳过] {img_path.name}：未检测到人脸")
                skipped += 1
                continue
            if len(locs) > 1:
                print(f"  [提示] {img_path.name}：检测到 {len(locs)} 张人脸，取最大的一张")
                # 取面积最大的人脸
                loc = max(locs, key=lambda t: (t[2]-t[0])*(t[1]-t[3]))
                enc = face_recognition.face_encodings(image, known_face_locations=[loc])[0]
            else:
                enc = face_recognition.face_encodings(image)[0]
            known_encodings.append(enc)
            known_names.append(name)
            print(f"  [注册] {name}  <-  {img_path.name}")
        except Exception as e:
            print(f"  [失败] {img_path.name}: {e}")
            skipped += 1

    if not known_encodings:
        print("[错误] 没有成功注册任何一张单人照片。")
        sys.exit(1)

    data = {"encodings": known_encodings, "names": known_names}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\n[完成] 共注册 {len(known_names)} 人，跳过 {skipped} 张。")
    return data


def recognize_one(image_path: Path, known: dict, roster: set = None):
    """对一张合照识别人脸并标注，返回统计信息。
    roster: 花名册姓名集合；若提供，则额外对比"名单外人员"。
    """
    roster = roster or set()
    image = face_recognition.load_image_file(str(image_path))
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        print(f"  {image_path.name}: 未检测到任何人脸。")
        return {"total": 0, "named": 0, "unknown": 0, "names": []}

    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

    known_encs = known["encodings"]
    known_names = known["names"]

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # 根据图片大小自适应标签字号
    font_size = max(16, int(min(pil_img.size) / 40))
    font = load_font(font_size)

    named_count = 0
    unknown_count = 0
    found_names = []

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encs, face_enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist <= TOLERANCE:
            name = known_names[best_idx]
            color = (0, 128, 0)      # 绿色：识别成功
            named_count += 1
            found_names.append(name)
        else:
            name = f"未知({best_dist:.2f})"
            color = (200, 0, 0)      # 红色：未识别
            unknown_count += 1

        # 画框
        line_w = max(2, int(min(pil_img.size) / 300))
        draw.rectangle([left, top, right, bottom], outline=color, width=line_w)

        # 标签背景
        text = name
        # Pillow >=10 用 textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 4
        label_y = bottom + 2
        draw.rectangle(
            [left, label_y, left + tw + pad * 2, label_y + th + pad * 2],
            fill=color,
        )
        draw.text((left + pad, label_y + pad), text, fill=(255, 255, 255), font=font)

    # 在图左上角写总人数
    total = len(face_locations)
    summary = f"共 {total} 人 | 识别出 {named_count} | 未识别 {unknown_count}"
    bbox = draw.textbbox((0, 0), summary, font=font)
    sw = bbox[2] - bbox[0]
    sh = bbox[3] - bbox[1]
    draw.rectangle([10, 10, 10 + sw + 16, 10 + sh + 16], fill=(0, 0, 0))
    draw.text((18, 16), summary, fill=(255, 255, 0), font=font)

    # 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S")
    out_path = OUTPUT_DIR / f"{image_path.stem}_标注_{ts}.jpg"
    pil_img.save(out_path, quality=92)

    print(f"  {image_path.name}: 共 {total} 人，识别出 {named_count}，未识别 {unknown_count}")
    if found_names:
        print(f"    已识别名单: {', '.join(found_names)}")

    # 与花名册对比
    outside = []   # 合照里识别到、但不在花名册里的人
    absent = []    # 花名册里有、但合照里没出现的人
    if roster:
        found_set = set(found_names)
        outside = sorted(found_set - roster)
        absent = sorted(roster - found_set)
        if outside:
            print(f"    >>> 名单外人员（{len(outside)} 人）: {', '.join(outside)}")
        else:
            print(f"    >>> 名单外人员: 无")
        if absent:
            print(f"    花名册未到场（{len(absent)} 人）: {', '.join(absent)}")

    return {
        "total": total,
        "named": named_count,
        "unknown": unknown_count,
        "names": found_names,
        "outside": outside,
        "absent": absent,
        "output": str(out_path),
    }


def main():
    print("=" * 60)
    print("合照标注 + 人数统计")
    print("=" * 60)

    if not TEST_DIR.exists():
        print(f"[错误] 未找到 test/ 目录：{TEST_DIR}")
        sys.exit(1)

    # 是否强制重建人脸库
    force = "--rebuild" in sys.argv

    known = build_known_faces(force=force)

    # 读取花名册
    roster = load_roster()
    if roster:
        print(f"\n[花名册] 已加载 {len(roster)} 个姓名：{', '.join(sorted(roster))}")
    else:
        print(f"\n[花名册] 未找到 {ROSTER_PATH.name} 或为空，将跳过名单对比。")

    # 扫描 test/ 下的合照
    photos = [p for p in sorted(TEST_DIR.iterdir())
              if p.is_file() and p.suffix.lower() in IMG_EXT]
    if not photos:
        print(f"\n[提示] test/ 目录下没有图片，请把合照放进去后重跑。")
        sys.exit(0)

    print(f"\n开始处理 {len(photos)} 张合照...\n")
    all_results = []
    for p in photos:
        r = recognize_one(p, known, roster=roster)
        r["file"] = p.name
        all_results.append(r)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['file']}: 总 {r['total']} 人，已识别 {r['named']}，未识别 {r['unknown']}")
        if r.get("outside"):
            print(f"    名单外人员: {', '.join(r['outside'])}")
        if r.get("absent"):
            print(f"    未到场: {', '.join(r['absent'])}")
        if r.get("output"):
            print(f"    标注图: {r['output']}")
    print()


if __name__ == "__main__":
    main()
