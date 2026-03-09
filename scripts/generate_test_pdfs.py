#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


class PDFTestFixtureGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.assets_dir = output_dir / "generated_assets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.font_name = self._register_chinese_font()
        self.styles = self._build_styles(self.font_name)

    def _register_chinese_font(self) -> str:
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("./fonts/NotoSansSC-Regular.ttf"),
            Path("./fonts/SourceHanSansSC-Regular.otf"),
        ]

        for idx, font_path in enumerate(candidates):
            if not font_path.exists():
                continue
            font_alias = f"CJKFont_{idx}"
            try:
                pdfmetrics.registerFont(TTFont(font_alias, str(font_path)))
                print(f"[INFO] Using Chinese font: {font_path}")
                return font_alias
            except Exception:
                try:
                    pdfmetrics.registerFont(TTFont(font_alias, str(font_path), subfontIndex=0))
                    print(f"[INFO] Using Chinese font (subfontIndex=0): {font_path}")
                    return font_alias
                except Exception as e:
                    print(f"[WARN] Failed to register font {font_path}: {e}")

        print(
            "[WARN] No usable Chinese TrueType font found. Falling back to Helvetica.\n"
            "Please place a Chinese .ttf/.otf under ./fonts if rendered text is garbled."
        )
        return "Helvetica"

    def _build_styles(self, font_name: str) -> dict:
        base = getSampleStyleSheet()
        return {
            "title": ParagraphStyle("TitleCN", parent=base["Title"], fontName=font_name, fontSize=18, leading=24, spaceAfter=14),
            "h2": ParagraphStyle("Heading2CN", parent=base["Heading2"], fontName=font_name, fontSize=14, leading=18, spaceBefore=8, spaceAfter=6),
            "body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName=font_name, fontSize=11, leading=18, spaceAfter=8),
            "caption": ParagraphStyle("CaptionCN", parent=base["BodyText"], fontName=font_name, fontSize=10, alignment=1, textColor=colors.darkslategray, spaceBefore=4, spaceAfter=10),
            "quote": ParagraphStyle("QuoteCN", parent=base["BodyText"], fontName=font_name, fontSize=11, leading=18, leftIndent=12, rightIndent=12, borderColor=colors.lightgrey, borderWidth=0.8, borderPadding=6, backColor=colors.whitesmoke, spaceBefore=6, spaceAfter=8),
        }

    @staticmethod
    def _create_dummy_image(path: Path, title: str, size: Tuple[int, int], bg: Tuple[int, int, int]) -> None:
        img = PILImage.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size[0] - text_w) // 2
        y = (size[1] - text_h) // 2
        draw.text((x, y), title, fill=(255, 255, 255), font=font)
        img.save(path)
        print(f"[INFO] Generated image: {path}")

    def generate_text_pdf(self) -> Path:
        pdf_path = self.output_dir / "test_chunking_text.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm)

        story = []
        story.append(Paragraph("人工智能在大语言模型中的应用综述", self.styles["title"]))
        story.append(Paragraph("基础概念", self.styles["h2"]))
        story.append(Paragraph("大语言模型（LLM）是一种基于深度学习的自然语言处理模型。它通过海量文本数据的预训练，学习语言的统计规律、语法结构甚至常识推理。", self.styles["body"]))
        story.append(Paragraph("RAG 技术的兴起", self.styles["h2"]))
        story.append(Paragraph("检索增强生成（Retrieval-Augmented Generation）解决了 LLM 的“幻觉”问题。其核心流程如下：", self.styles["body"]))

        bullet_items = [
            "加载：读取不同格式的文档。",
            "切片 (Chunking)：将长文档切分为固定大小或按语义划分的块。",
            "向量化：将文字转为 Embedding 向量。",
            "检索：根据用户问题匹配最相似的文本块。",
        ]
        story.append(ListFlowable([ListItem(Paragraph(item, self.styles["body"])) for item in bullet_items], bulletType="bullet", leftIndent=16))
        story.append(Spacer(1, 8))

        story.append(Paragraph("切片策略的重要性", self.styles["h2"]))
        story.append(Paragraph("“如果 Chunk 太小，会丢失上下文；如果 Chunk 太大，会引入过多噪音并超出模型窗口。”", self.styles["quote"]))
        doc.build(story)
        print(f"[INFO] Generated PDF: {pdf_path}")
        return pdf_path

    def generate_multimodal_pdf(self) -> Path:
        img1_path = self.assets_dir / "architecture_diagram.png"
        img2_path = self.assets_dir / "sharding_diagram.png"
        self._create_dummy_image(img1_path, "Architecture Diagram", (1200, 700), (58, 94, 146))
        self._create_dummy_image(img2_path, "Sharding Diagram", (1200, 700), (80, 130, 90))

        pdf_path = self.output_dir / "test_chunking_multimodal.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm)
        story = []
        story.append(Paragraph("现代分布式系统架构指南", self.styles["title"]))
        story.append(Paragraph("系统架构概述", self.styles["h2"]))
        story.append(Paragraph("分布式系统通过将计算与存储能力分散到多个节点，实现高可用、可扩展与容错。常见设计包括服务拆分、异步消息、负载均衡与故障转移机制。", self.styles["body"]))
        story.append(RLImage(str(img1_path), width=15 * cm, height=8.5 * cm))
        story.append(Paragraph("图 1：分布式系统高层架构", self.styles["caption"]))

        story.append(Paragraph("数据库分片策略", self.styles["h2"]))
        story.append(Paragraph("水平分片按行拆分数据以分摊读写压力，适合大规模同构数据；垂直分片按字段或业务域拆分，适合隔离热点模块并降低耦合。", self.styles["body"]))
        story.append(RLImage(str(img2_path), width=15 * cm, height=8.5 * cm))
        story.append(Paragraph("图 2：水平分片与垂直分片对比", self.styles["caption"]))

        doc.build(story)
        print(f"[INFO] Generated PDF: {pdf_path}")
        return pdf_path


def main() -> int:
    out_dir = Path("tests/data")
    generator = PDFTestFixtureGenerator(output_dir=out_dir)
    p1 = generator.generate_text_pdf()
    p2 = generator.generate_multimodal_pdf()
    print("[DONE]", p1, p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
