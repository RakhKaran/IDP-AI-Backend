import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional


Logger = Callable[[str], None]


@dataclass(frozen=True)
class ConversionResult:
    input_path: str
    pipeline_pdf_path: Optional[str]
    converted: bool
    reason: str = ""


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_DOC_EXTS = {".docx", ".doc"}


def is_supported_input(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".pdf", *_IMAGE_EXTS, *_DOC_EXTS}


def normalize_to_pipeline_pdf(
    input_path: str,
    *,
    output_dir: str,
    logger: Optional[Logger] = None,
    overwrite: bool = False,
    suffix: str = "__pipeline",
    timeout_seconds: int = 120,
) -> ConversionResult:
    """
    Ensure `input_path` is represented as a PDF for downstream pipeline steps.

    - PDFs are returned unchanged.
    - Images are converted to a single-page PDF via Pillow.
    - DOC/DOCX are converted to PDF using LibreOffice (`soffice`) if available.

    Returns a ConversionResult; it never raises for expected "tool not available"
    situations (it will return converted=False with a reason).
    """
    log = logger or (lambda _msg: None)
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    stem = os.path.splitext(os.path.basename(input_path))[0]

    if ext == ".pdf":
        return ConversionResult(input_path=input_path, pipeline_pdf_path=input_path, converted=False, reason="already_pdf")

    pipeline_pdf_path = os.path.join(output_dir, f"{stem}{suffix}.pdf")
    if os.path.exists(pipeline_pdf_path) and not overwrite:
        return ConversionResult(
            input_path=input_path,
            pipeline_pdf_path=pipeline_pdf_path,
            converted=True,
            reason="already_converted",
        )

    try:
        if ext in _IMAGE_EXTS:
            return _convert_image_to_pdf(input_path, pipeline_pdf_path, log)

        if ext in _DOC_EXTS:
            return _convert_office_to_pdf(
                input_path,
                output_dir=output_dir,
                expected_pdf_path=pipeline_pdf_path,
                log=log,
                timeout_seconds=timeout_seconds,
                suffix=suffix,
            )

        return ConversionResult(
            input_path=input_path,
            pipeline_pdf_path=None,
            converted=False,
            reason=f"unsupported_extension:{ext}",
        )
    except Exception as exc:
        return ConversionResult(
            input_path=input_path,
            pipeline_pdf_path=None,
            converted=False,
            reason=f"conversion_failed:{type(exc).__name__}:{exc}",
        )


def _convert_image_to_pdf(image_path: str, out_pdf_path: str, log: Logger) -> ConversionResult:
    from PIL import Image

    with Image.open(image_path) as img:
        if getattr(img, "is_animated", False):
            frames = []
            try:
                i = 0
                while True:
                    img.seek(i)
                    frame = img.convert("RGB")
                    frames.append(frame)
                    i += 1
            except EOFError:
                pass

            if not frames:
                frames = [img.convert("RGB")]
            first, rest = frames[0], frames[1:]
            first.save(out_pdf_path, "PDF", save_all=True, append_images=rest)
        else:
            rgb = img.convert("RGB")
            rgb.save(out_pdf_path, "PDF")

    log(f"Converted image -> PDF: {os.path.basename(image_path)} -> {os.path.basename(out_pdf_path)}")
    return ConversionResult(input_path=image_path, pipeline_pdf_path=out_pdf_path, converted=True, reason="image_to_pdf")


def _convert_office_to_pdf(
    office_path: str,
    *,
    output_dir: str,
    expected_pdf_path: str,
    log: Logger,
    timeout_seconds: int,
    suffix: str,
) -> ConversionResult:
    soffice = shutil.which("soffice")
    if not soffice:
        return ConversionResult(
            input_path=office_path,
            pipeline_pdf_path=None,
            converted=False,
            reason="soffice_not_found",
        )

    base_stem = os.path.splitext(os.path.basename(office_path))[0]
    default_pdf_path = os.path.join(output_dir, f"{base_stem}.pdf")

    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        output_dir,
        office_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        return ConversionResult(
            input_path=office_path,
            pipeline_pdf_path=None,
            converted=False,
            reason=f"soffice_failed:rc={proc.returncode}:stdout={stdout}:stderr={stderr}",
        )

    if os.path.exists(expected_pdf_path):
        log(f"Converted office -> PDF: {os.path.basename(office_path)} -> {os.path.basename(expected_pdf_path)}")
        return ConversionResult(
            input_path=office_path,
            pipeline_pdf_path=expected_pdf_path,
            converted=True,
            reason="office_to_pdf",
        )

    if os.path.exists(default_pdf_path):
        renamed = os.path.join(output_dir, f"{base_stem}{suffix}.pdf")
        try:
            os.replace(default_pdf_path, renamed)
        except Exception:
            renamed = default_pdf_path

        log(f"Converted office -> PDF: {os.path.basename(office_path)} -> {os.path.basename(renamed)}")
        return ConversionResult(
            input_path=office_path,
            pipeline_pdf_path=renamed,
            converted=True,
            reason="office_to_pdf",
        )

    return ConversionResult(
        input_path=office_path,
        pipeline_pdf_path=None,
        converted=False,
        reason="soffice_no_output_pdf",
    )

