"""C3: PdfLoader 契约测试。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.libs.loader.pdf_loader import PdfLoader


class _FakeImage:
    """测试用图片对象：模拟 pypdf 的图片结构。"""

    def __init__(self, data: bytes, width: int = 100, height: int = 80) -> None:
        self.data = data
        self.width = width
        self.height = height


class _FakePage:
    """测试用 page 对象：可配置文本和图片输出。"""

    def __init__(self, text: str, images: list[object] | None = None) -> None:
        self._text = text
        self._images = images or []

    def extract_text(self) -> str:
        return self._text

    @property
    def images(self) -> list[object]:
        return self._images


class _BrokenImagePage(_FakePage):
    """测试用 page：访问 images 时抛错，验证降级行为。"""

    @property
    def images(self) -> list[object]:
        raise RuntimeError("image extraction failed")


class _FakeReader:
    """测试用 reader：仅暴露 pages 字段即可。"""

    def __init__(self, pages: list[object]) -> None:
        self.pages = pages


def test_pdf_loader_returns_document_for_text_only_pdf(monkeypatch, tmp_path: Path) -> None:
    sample_pdf = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"
    loader = PdfLoader(images_root=str(tmp_path / "images"))

    fake_reader = _FakeReader(
        pages=[
            _FakePage("第一页：这是纯文本内容。"),
            _FakePage("第二页：继续文本内容。"),
        ]
    )
    monkeypatch.setattr(loader, "_open_reader", lambda _path: fake_reader)

    doc = loader.load(str(sample_pdf))
    print(f"[C3] text-only doc.id={doc.id} source={doc.metadata['source_path']}")
    print(f"[C3] text-only doc.text={doc.text}")

    assert doc.metadata["source_path"] == str(sample_pdf)
    assert "第一页：这是纯文本内容。" in doc.text
    assert "第二页：继续文本内容。" in doc.text
    assert doc.metadata.get("images", []) == []


def test_pdf_loader_extracts_images_and_inserts_placeholders(monkeypatch, tmp_path: Path) -> None:
    sample_pdf = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "with_images.pdf"
    loader = PdfLoader(images_root=str(tmp_path / "images"))

    fake_png = b"\x89PNG\r\n\x1a\nfake-image"
    fake_reader = _FakeReader(
        pages=[
            _FakePage("这一页包含图片。", images=[_FakeImage(data=fake_png, width=320, height=200)]),
        ]
    )
    monkeypatch.setattr(loader, "_open_reader", lambda _path: fake_reader)

    doc = loader.load(str(sample_pdf))
    images = doc.metadata.get("images", [])
    print(f"[C3] image doc.id={doc.id}")
    print(f"[C3] image placeholders text={doc.text}")
    print(f"[C3] image metadata={images}")

    assert len(images) == 1
    image_meta = images[0]
    placeholder = f"[IMAGE: {image_meta['id']}]"
    assert placeholder in doc.text
    assert image_meta["text_length"] == len(placeholder)
    assert image_meta["text_offset"] == doc.text.index(placeholder)
    assert Path(image_meta["path"]).exists()
    assert image_meta["position"]["width"] == 320
    assert image_meta["position"]["height"] == 200


def test_pdf_loader_degrades_gracefully_when_image_extraction_fails(monkeypatch, tmp_path: Path) -> None:
    sample_pdf = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "with_images.pdf"
    loader = PdfLoader(images_root=str(tmp_path / "images"))

    fake_reader = _FakeReader(pages=[_BrokenImagePage("文本应当仍然可用。")])
    monkeypatch.setattr(loader, "_open_reader", lambda _path: fake_reader)

    doc = loader.load(str(sample_pdf))
    print(f"[C3] degraded doc.text={doc.text}")
    print(f"[C3] degraded doc.images={doc.metadata.get('images', [])}")

    assert "文本应当仍然可用。" in doc.text
    assert doc.metadata.get("images", []) == []
