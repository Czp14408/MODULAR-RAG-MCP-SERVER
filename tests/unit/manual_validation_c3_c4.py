"""C3 + C4 手动验收脚本（统一放在 tests/unit）。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.pdf_loader import PdfLoader


def manual_validate_c3_pdf_loader() -> bool:
    """手动验收 C3：验证 PdfLoader 真实解析行为。"""
    text_pdf = PROJECT_ROOT / "tests" / "data" / "test_chunking_text.pdf"
    multimodal_pdf = PROJECT_ROOT / "tests" / "data" / "test_chunking_multimodal.pdf"
    image_output_root = PROJECT_ROOT / "tests" / "data" / "manual_images"

    print("[C3-MANUAL] ===== 手动验收开始 =====")
    print(f"[C3-MANUAL] text_pdf={text_pdf}")
    print(f"[C3-MANUAL] multimodal_pdf={multimodal_pdf}")
    print(f"[C3-MANUAL] image_output_root={image_output_root}")

    if not text_pdf.exists() or not multimodal_pdf.exists():
        print("[C3-MANUAL][FAIL] 缺少测试 PDF，请先执行 scripts/generate_test_pdfs.py")
        return False

    loader = PdfLoader(images_root=str(image_output_root))
    failures: List[str] = []

    doc_text: Document = loader.load(str(text_pdf))
    print("\n[C3-MANUAL] --- 纯文本 PDF 结果 ---")
    print(f"[C3-MANUAL] doc.id={doc_text.id}")
    print(f"[C3-MANUAL] metadata.source_path={doc_text.metadata.get('source_path')}")
    print(f"[C3-MANUAL] metadata.images={doc_text.metadata.get('images', [])}")
    print(f"[C3-MANUAL] text.preview={doc_text.text[:180].replace(chr(10), ' ')}")

    if doc_text.metadata.get("source_path") != str(text_pdf):
        failures.append("纯文本 PDF 的 metadata.source_path 不正确")
    if not isinstance(doc_text.text, str) or len(doc_text.text.strip()) == 0:
        failures.append("纯文本 PDF 未解析出有效文本")
    if doc_text.metadata.get("images", []) not in ([], None):
        failures.append("纯文本 PDF 的 metadata.images 应为空列表或省略")

    doc_multi: Document = loader.load(str(multimodal_pdf))
    images = doc_multi.metadata.get("images", [])
    print("\n[C3-MANUAL] --- 多模态 PDF 结果 ---")
    print(f"[C3-MANUAL] doc.id={doc_multi.id}")
    print(f"[C3-MANUAL] metadata.source_path={doc_multi.metadata.get('source_path')}")
    print(f"[C3-MANUAL] images.count={len(images)}")
    print(f"[C3-MANUAL] text.preview={doc_multi.text[:220].replace(chr(10), ' ')}")

    if doc_multi.metadata.get("source_path") != str(multimodal_pdf):
        failures.append("多模态 PDF 的 metadata.source_path 不正确")
    if not isinstance(images, list):
        failures.append("多模态 PDF 的 metadata.images 不是 list")
    if len(images) == 0:
        failures.append("多模态 PDF 未提取到图片（images.count=0）")

    for idx, image_meta in enumerate(images):
        print(f"[C3-MANUAL] image[{idx}]={image_meta}")
        required_keys = {"id", "path", "page", "text_offset", "text_length"}
        if not required_keys.issubset(set(image_meta.keys())):
            failures.append(f"image[{idx}] 缺少必需字段: {required_keys - set(image_meta.keys())}")
            continue

        placeholder = f"[IMAGE: {image_meta['id']}]"
        if placeholder not in doc_multi.text:
            failures.append(f"image[{idx}] 的占位符未写入 Document.text")
        else:
            actual_offset = doc_multi.text.index(placeholder)
            actual_length = len(placeholder)
            if image_meta["text_offset"] != actual_offset:
                failures.append(
                    f"image[{idx}] text_offset 不匹配: expected={actual_offset}, actual={image_meta['text_offset']}"
                )
            if image_meta["text_length"] != actual_length:
                failures.append(
                    f"image[{idx}] text_length 不匹配: expected={actual_length}, actual={image_meta['text_length']}"
                )

        if not Path(image_meta["path"]).exists():
            failures.append(f"image[{idx}] 文件未落盘: {image_meta['path']}")

    print("\n[C3-MANUAL] ===== 验收结论 =====")
    if failures:
        print("[C3-MANUAL][FAIL] 发现以下问题：")
        for item in failures:
            print(f"  - {item}")
        return False
    print("[C3-MANUAL][PASS] 所有 C3 验收项均通过。")
    return True


def manual_validate_c4_document_chunker() -> bool:
    """手动验收 C4：验证 DocumentChunker 适配层行为。"""
    failures: List[str] = []
    print("\n[C4-MANUAL] ===== 手动验收开始 =====")

    document = Document(
        id="doc_c4_manual",
        text="第一段：这是用于手动验收的长文本。" * 5 + "\n\n" + "第二段：继续补充内容。" * 5,
        metadata={
            "source_path": "tests/data/test_chunking_text.pdf",
            "doc_type": "pdf",
            "title": "C4 手动验收文档",
        },
    )

    # 参数选择说明：
    # 1) 固定使用 fixed splitter，保证手动验收时结果稳定可复现。
    # 2) 通过不同 chunk_size 对比，验证“配置驱动”是否生效。
    cfg_small = {"splitter": {"provider": "fixed", "chunk_size": 30}}
    cfg_large = {"splitter": {"provider": "fixed", "chunk_size": 90}}

    chunker_small = DocumentChunker(cfg_small)
    chunker_large = DocumentChunker(cfg_large)

    chunks_small = chunker_small.split_document(document)
    chunks_small_again = chunker_small.split_document(document)
    chunks_large = chunker_large.split_document(document)

    ids_1 = [c.id for c in chunks_small]
    ids_2 = [c.id for c in chunks_small_again]
    print(f"[C4-MANUAL] chunks_small={len(chunks_small)} chunks_large={len(chunks_large)}")
    print(f"[C4-MANUAL] ids_run1={ids_1}")
    print(f"[C4-MANUAL] ids_run2={ids_2}")

    if len(chunks_small) <= len(chunks_large):
        failures.append("配置驱动失败：chunk_size 更小却没有产生更多 chunks")

    if len(set(ids_1)) != len(ids_1):
        failures.append("ID 唯一性失败：存在重复 chunk id")
    if ids_1 != ids_2:
        failures.append("ID 确定性失败：同一文档重复切分得到不同 id 序列")

    for i, chunk in enumerate(chunks_small):
        print(
            f"[C4-MANUAL] chunk[{i}] id={chunk.id} "
            f"source_ref={chunk.source_ref} idx={chunk.metadata.get('chunk_index')} "
            f"offset=({chunk.start_offset},{chunk.end_offset})"
        )
        if chunk.source_ref != document.id:
            failures.append(f"chunk[{i}] source_ref 未指向父 document.id")
        if chunk.metadata.get("source_path") != document.metadata.get("source_path"):
            failures.append(f"chunk[{i}] metadata.source_path 未继承")
        if chunk.metadata.get("chunk_index") != i:
            failures.append(f"chunk[{i}] chunk_index 不正确")
        if chunk.end_offset < chunk.start_offset:
            failures.append(f"chunk[{i}] offset 非法")

    print("\n[C4-MANUAL] ===== 验收结论 =====")
    if failures:
        print("[C4-MANUAL][FAIL] 发现以下问题：")
        for item in failures:
            print(f"  - {item}")
        return False
    print("[C4-MANUAL][PASS] 所有 C4 验收项均通过。")
    return True


if __name__ == "__main__":
    # c3_ok = manual_validate_c3_pdf_loader()
    c4_ok = manual_validate_c4_document_chunker()
    print(f"\n[MANUAL-SUMMARY] C3={c3_ok} C4={c4_ok}")
    raise SystemExit(0 if (c3_ok and c4_ok) else 1)
