# 合照标注 + 花名册比对工具

基于开源库 [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)（56.7k stars）搭建的本地工具。

## 功能

- 扫描合照，自动检测人脸并在脸上画框、标注姓名；
- 图左上角自动显示"共 N 人 | 识别出 X | 未识别 Y"；
- 与本地花名册（`roster.txt`）比对，**单独列出"名单外人员"**（合照里出现了但名单里没有的人）；
- 同时列出"未到场"（名单里有但合照里没出现的人）。

> 所有数据均在本地运行，不上传任何图片、姓名或人脸特征。

## 目录结构

```
StuNum/
├── known/              # 单人照片库（不上传）：每人一张，文件名=姓名
├── test/              # 待标注合照（不上传）
├── output/            # 标注结果图（不上传）
├── roster.txt         # 花名册，每行一个姓名（不上传，参考 roster.example.txt）
├── mark_photo.py      # 主脚本
├── 一键标注.bat       # Windows 双击运行
└── .venv/             # Python 虚拟环境（不上传）
```

## 快速开始（Windows）

### 1. 环境要求

- Python 3.12（3.13/3.14 上 dlib 可能没有预编译包）
- 依赖：`pip install face_recognition scikit-learn Pillow "setuptools<81"`

> 注意：`face_recognition_models` 依赖 `pkg_resources`，需 `setuptools<81`。

### 2. 准备数据

```powershell
# 单人照片库：known\张三.jpg, known\李四.jpg ...
mkdir known, test, output

# 花名册：复制模板后填写
cp roster.example.txt roster.txt
notepad roster.txt
```

### 3. 运行

```powershell
# 首次运行自动注册 known/ 里的人脸
.venv\Scripts\python.exe mark_photo.py

# 加了新人后强制重建人脸库
.venv\Scripts\python.exe mark_photo.py --rebuild
```

或直接双击 `一键标注.bat`。

## 输出示例

```
============================================================
 合照标注 + 人数统计
============================================================
[花名册] 已加载 35 个姓名：王一, 王二, ...

开始处理 1 张合照...

  班级合影.jpg: 共 35 人，识别出 33，未识别 2
    已识别名单: 王一, 王二, ...
    >>> 名单外人员（1 人）: 校外辅导员
    花名册未到场（2 人）: 张三, 李四

    标注图: D:\...\output\班级合影_标注_103045.jpg
```

## 可调参数

在 `mark_photo.py` 顶部：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TOLERANCE` | 0.55 | 人脸识别阈值，越小越严格（误报减少，漏认增加） |

## 隐私声明

本仓库**不包含**任何学生照片、花名册真实数据或人脸特征文件。
`known/`、`test/`、`output/`、`roster.txt`、`.venv/`、`known_faces.pkl` 均已在 `.gitignore` 中排除。

## 致谢

- [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) —— 核心人脸识别库
