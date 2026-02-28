import os
import tempfile
import unittest
import zipfile

from PIL import Image

from core.proof_pack import export_proof_pack
from core.report import build_stegano_report, render_presentation_summary, render_report_text


class ReportSchemaTests(unittest.TestCase):
    def test_schema_1_1_contains_competition_fields(self):
        report = build_stegano_report(
            source_image_path="/tmp/source.png",
            image_size=(320, 240),
            method="sequential",
            bits_per_channel=1,
            password_used=False,
            message_chars=10,
            message_bytes_utf8=10,
            capacity_bytes=100,
            psnr_db=90.0,
            mse=0.0,
            ssim=1.0,
            metrics_error=None,
            chi_original={"chi2": 1.2, "zero_bits": 10, "one_bits": 11, "suspicious": False},
            chi_stego={"chi2": 2.3, "zero_bits": 10, "one_bits": 11, "suspicious": False},
            risk_level="LOW",
            risk_reason="низкая загрузка",
            demo_summary={"completed": True},
            robustness_score=80.0,
            visual_artifacts={"heatmap": "proof_pack/heatmap.png"},
            recommendation="ok",
        )
        self.assertEqual(report["schema"]["version"], "1.1.0")
        self.assertIn("demo_summary", report)
        self.assertIn("robustness_score", report)
        self.assertIn("visual_artifacts", report)
        self.assertIn("recommendation", report)

        txt = render_report_text(report)
        self.assertIn("Дополнительные поля 1.1", txt)
        self.assertIn("Скриншот hotspot", txt)
        summary = render_presentation_summary(report)
        self.assertIn("Краткая сводка", summary)

    def test_proof_pack_export_contains_all_artifacts(self):
        source = Image.new("RGB", (64, 64), (100, 120, 140))
        encoded = Image.new("RGB", (64, 64), (101, 121, 141))
        report = build_stegano_report(
            source_image_path="source.png",
            image_size=source.size,
            method="sequential",
            bits_per_channel=1,
            password_used=False,
            message_chars=2,
            message_bytes_utf8=2,
            capacity_bytes=100,
            psnr_db=50.0,
            mse=0.5,
            ssim=0.99,
            metrics_error=None,
            chi_original={"chi2": 1.2, "zero_bits": 10, "one_bits": 11, "suspicious": False},
            chi_stego={"chi2": 2.3, "zero_bits": 10, "one_bits": 11, "suspicious": False},
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "proof_pack.zip")
            export_proof_pack(
                zip_path,
                report,
                source,
                encoded,
                attacks=[],
                extra_png_artifacts={
                    "proof_pack/hotspot.png": b"fake-hotspot",
                    "proof_pack/inspector.png": b"fake-inspector",
                },
            )
            self.assertTrue(os.path.exists(zip_path))
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
            self.assertEqual(
                names,
                {
                    "proof_pack/report.json",
                    "proof_pack/report.txt",
                    "proof_pack/before.png",
                    "proof_pack/after.png",
                    "proof_pack/heatmap.png",
                    "proof_pack/attacks.csv",
                    "proof_pack/hotspot.png",
                    "proof_pack/inspector.png",
                },
            )

            with open(zip_path, "rb") as f1:
                first_bytes = f1.read()
            export_proof_pack(
                zip_path,
                report,
                source,
                encoded,
                attacks=[],
                extra_png_artifacts={
                    "proof_pack/hotspot.png": b"fake-hotspot",
                    "proof_pack/inspector.png": b"fake-inspector",
                },
            )
            with open(zip_path, "rb") as f2:
                second_bytes = f2.read()
            self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
