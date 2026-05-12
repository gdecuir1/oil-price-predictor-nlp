"""
Comprehensive Report Generator
================================

Produces detailed prediction reports in JSON and HTML formats from
:class:`PredictionResult` objects.  The report includes:

* **Executive summary** — primary market direction prediction with
  confidence score and uncertainty estimate.
* **Sub-domain breakdown** — detailed analysis for each auxiliary
  prediction task (magnitude, timeframe, volatility, sentiment,
  supply/demand impact, geopolitical risk).
* **Per-article analysis** — how each individual article contributed
  to the overall prediction.
* **Confidence diagnostics** — flags low-confidence predictions and
  high-uncertainty estimates.
* **Methodology notes** — explains the model architecture and
  prediction methodology for transparency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config, SUBDOMAIN_SPECS
from .predictor import PredictionResult, ArticlePrediction

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive prediction reports.

    Args:
        config: Model configuration for sub-domain metadata.
        output_dir: Directory to write reports to.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Set up the report generator."""
        self.config = config or Config()
        self.output_dir = output_dir or self.config.inference.report_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: PredictionResult,
        report_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Generate all configured report formats.

        Args:
            result: Aggregated prediction result from the predictor.
            report_name: Optional base name for report files.
                Defaults to a timestamped name.

        Returns:
            Dict mapping format names to output file paths.
        """
        if report_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_name = f"oil_market_prediction_{timestamp}"

        output_paths: Dict[str, Path] = {}

        # Generate JSON report
        if self.config.inference.generate_json_report:
            json_path = self.output_dir / f"{report_name}.json"
            self._write_json_report(result, json_path)
            output_paths["json"] = json_path

        # Generate HTML report
        if self.config.inference.generate_html_report:
            html_path = self.output_dir / f"{report_name}.html"
            self._write_html_report(result, html_path)
            output_paths["html"] = html_path

        logger.info("Reports generated: %s", output_paths)
        return output_paths

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def _write_json_report(self, result: PredictionResult, path: Path) -> None:
        """Write a machine-readable JSON prediction report.

        Args:
            result: Prediction result object.
            path: Output file path.
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "1.0.0",
                "n_articles_processed": result.n_articles_processed,
                "n_articles_total": result.n_articles_total,
                "extraction_rate": (
                    result.n_articles_processed / max(result.n_articles_total, 1)
                ),
            },
            "executive_summary": {
                "primary_prediction": result.primary_prediction,
                "primary_confidence": round(result.primary_confidence, 4),
                "primary_probabilities": {
                    k: round(v, 4) for k, v in result.primary_probabilities.items()
                },
                "uncertainty_estimate": round(result.ensemble_std, 4),
                "confidence_flag": self._confidence_flag(result.primary_confidence),
            },
            "subdomain_analysis": {},
            "per_article_predictions": [],
            "methodology": self._methodology_section(),
        }

        # Sub-domain analysis
        for key in sorted(result.subdomain_aggregations.keys()):
            agg = result.subdomain_aggregations[key]
            spec = SUBDOMAIN_SPECS.get(key)

            report["subdomain_analysis"][key] = {
                "name": spec.name if spec else key,
                "description": spec.description if spec else "",
                "prediction": agg["prediction"],
                "confidence": round(agg["confidence"], 4),
                "uncertainty": round(agg.get("uncertainty", 0), 4),
                "probabilities": {
                    k: round(v, 4) for k, v in agg["probabilities"].items()
                },
                "n_articles_contributing": agg.get("n_articles", 0),
            }

        # Per-article predictions (limited to top-K most influential)
        sorted_articles = sorted(
            result.article_predictions,
            key=lambda a: a.confidence,
            reverse=True,
        )
        for article_pred in sorted_articles[: self.config.inference.top_k_articles]:
            article_entry = {
                "filename": article_pred.filename,
                "title": article_pred.title,
                "source": article_pred.source,
                "confidence": round(article_pred.confidence, 4),
                "uncertainty": round(article_pred.uncertainty, 4),
                "predictions": {},
            }
            for sk, sp in article_pred.subdomain_predictions.items():
                article_entry["predictions"][sk] = {
                    "label": sp["predicted_label"],
                    "confidence": round(sp["confidence"], 4),
                }
            report["per_article_predictions"].append(article_entry)

        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("JSON report written to %s", path)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _write_html_report(self, result: PredictionResult, path: Path) -> None:
        """Write a styled HTML prediction report.

        The report uses inline CSS for portability — no external
        dependencies required to view it in a browser.

        Args:
            result: Prediction result object.
            path: Output file path.
        """
        # Direction → colour mapping for visual cues
        direction_colours = {
            "up": "#22c55e",       # green
            "unchanged": "#f59e0b",  # amber
            "down": "#ef4444",     # red
        }
        primary_colour = direction_colours.get(
            result.primary_prediction, "#6b7280"
        )

        confidence_pct = result.primary_confidence * 100
        flag = self._confidence_flag(result.primary_confidence)

        # Build sub-domain rows
        subdomain_rows = ""
        for key in sorted(result.subdomain_aggregations.keys()):
            agg = result.subdomain_aggregations[key]
            spec = SUBDOMAIN_SPECS.get(key)
            name = spec.name if spec else key

            # Build probability bar chart
            prob_bars = ""
            for label, prob in sorted(agg["probabilities"].items(), key=lambda x: -x[1]):
                bar_width = prob * 100
                prob_bars += (
                    f'<div style="display:flex;align-items:center;margin:2px 0;">'
                    f'<span style="width:120px;font-size:12px;">{label}</span>'
                    f'<div style="background:#e5e7eb;border-radius:4px;flex:1;height:16px;">'
                    f'<div style="background:#3b82f6;border-radius:4px;height:100%;'
                    f'width:{bar_width:.1f}%;"></div></div>'
                    f'<span style="width:50px;text-align:right;font-size:12px;">'
                    f'{prob:.1%}</span></div>'
                )

            is_primary = key == "market_direction"
            section_style = (
                "border-left:4px solid #3b82f6;padding-left:12px;"
                if is_primary else ""
            )

            subdomain_rows += f"""
            <div style="margin-bottom:20px;{section_style}">
                <h3 style="margin:0 0 4px 0;color:#1f2937;">
                    {name} {'⭐' if is_primary else ''}
                </h3>
                <p style="color:#6b7280;font-size:13px;margin:0 0 8px 0;">
                    {spec.description if spec else ''}
                </p>
                <div style="display:flex;gap:16px;margin-bottom:8px;">
                    <div>
                        <strong>Prediction:</strong>
                        <span style="color:{primary_colour if is_primary else '#1f2937'};">
                            {agg['prediction'].upper()}
                        </span>
                    </div>
                    <div>
                        <strong>Confidence:</strong> {agg['confidence']:.1%}
                    </div>
                    <div>
                        <strong>Uncertainty:</strong> {agg.get('uncertainty', 0):.4f}
                    </div>
                </div>
                <div style="max-width:500px;">{prob_bars}</div>
            </div>
            """

        # Build per-article table rows
        article_rows = ""
        sorted_articles = sorted(
            result.article_predictions,
            key=lambda a: a.confidence,
            reverse=True,
        )
        for i, ap in enumerate(sorted_articles[: self.config.inference.top_k_articles]):
            primary_pred = ap.subdomain_predictions.get("market_direction", {})
            pred_label = primary_pred.get("predicted_label", "N/A")
            pred_conf = primary_pred.get("confidence", 0)

            article_rows += f"""
            <tr>
                <td>{i + 1}</td>
                <td title="{ap.filename}">{ap.title or ap.filename[:40]}</td>
                <td>{ap.source or 'Unknown'}</td>
                <td><strong>{pred_label.upper()}</strong></td>
                <td>{pred_conf:.1%}</td>
                <td>{ap.uncertainty:.4f}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oil Market Prediction Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            background: #f9fafb; color: #1f2937; line-height: 1.6;
            padding: 24px; max-width: 1000px; margin: 0 auto;
        }}
        h1 {{ font-size: 24px; margin-bottom: 4px; }}
        h2 {{ font-size: 18px; color: #374151; margin: 24px 0 12px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
        .card {{
            background: white; border-radius: 8px; padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
        }}
        .hero {{
            text-align: center; padding: 32px;
            background: linear-gradient(135deg, #1e3a5f, #2563eb);
            color: white; border-radius: 12px; margin-bottom: 24px;
        }}
        .hero .direction {{ font-size: 48px; font-weight: 700; letter-spacing: 2px; }}
        .hero .conf {{ font-size: 18px; opacity: 0.9; }}
        .flag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
        .flag-high {{ background: #dcfce7; color: #166534; }}
        .flag-medium {{ background: #fef9c3; color: #854d0e; }}
        .flag-low {{ background: #fee2e2; color: #991b1b; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        .meta {{ color: #6b7280; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="hero">
        <p class="meta" style="color:rgba(255,255,255,0.7);">
            US Oil Market Prediction Report
        </p>
        <div class="direction" style="color:{primary_colour};">
            {result.primary_prediction.upper()}
        </div>
        <div class="conf">
            Confidence: {confidence_pct:.1f}%
            <span class="flag flag-{flag.lower()}">{flag}</span>
        </div>
        <p class="meta" style="color:rgba(255,255,255,0.6);margin-top:8px;">
            Based on analysis of {result.n_articles_processed} articles
            (of {result.n_articles_total} total)
            | Uncertainty: {result.ensemble_std:.4f}
        </p>
    </div>

    <h2>Primary Prediction Probabilities</h2>
    <div class="card">
        <div style="display:flex;gap:24px;justify-content:center;">
            <div style="text-align:center;">
                <div style="font-size:28px;color:#22c55e;font-weight:700;">
                    {result.primary_probabilities.get('up', 0):.1%}
                </div>
                <div style="font-size:13px;color:#6b7280;">UP</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;color:#f59e0b;font-weight:700;">
                    {result.primary_probabilities.get('unchanged', 0):.1%}
                </div>
                <div style="font-size:13px;color:#6b7280;">UNCHANGED</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;color:#ef4444;font-weight:700;">
                    {result.primary_probabilities.get('down', 0):.1%}
                </div>
                <div style="font-size:13px;color:#6b7280;">DOWN</div>
            </div>
        </div>
    </div>

    <h2>Sub-Domain Analysis</h2>
    <div class="card">
        {subdomain_rows}
    </div>

    <h2>Top Contributing Articles</h2>
    <div class="card" style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Article</th>
                    <th>Source</th>
                    <th>Direction</th>
                    <th>Confidence</th>
                    <th>Uncertainty</th>
                </tr>
            </thead>
            <tbody>
                {article_rows}
            </tbody>
        </table>
    </div>

    <h2>Methodology</h2>
    <div class="card">
        <p style="font-size:14px;color:#4b5563;">
            This report was generated by a multi-task transformer neural network
            trained on scraped oil market news articles.  The model uses a
            {self.config.model.pretrained_model_name if self.config.model.use_pretrained else 'custom'}
            encoder backbone with separate classification heads for each
            prediction sub-domain.  Uncertainty estimates are produced via
            MC-Dropout ({self.config.inference.ensemble_passes} forward passes).
        </p>
        <p style="font-size:12px;color:#9ca3af;margin-top:8px;">
            Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        </p>
    </div>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", path)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_flag(confidence: float) -> str:
        """Classify confidence level for visual flagging.

        Args:
            confidence: Prediction confidence (0-1).

        Returns:
            ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        """
        if confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _methodology_section() -> Dict[str, str]:
        """Return methodology notes for the JSON report.

        Returns:
            Dict with methodology description fields.
        """
        return {
            "model_type": "Multi-task Transformer (FinBERT-based)",
            "input": "Raw HTML news articles about US oil markets",
            "primary_output": (
                "Market direction classification: UP (prices expected to rise), "
                "UNCHANGED (no significant movement), DOWN (prices expected to fall)"
            ),
            "auxiliary_outputs": (
                "Price magnitude, timeframe, volatility, sentiment, "
                "supply impact, demand impact, geopolitical risk"
            ),
            "uncertainty_method": "MC-Dropout (Monte Carlo Dropout) ensemble",
            "aggregation": (
                "Confidence-weighted probability averaging across all articles"
            ),
            "limitations": (
                "Predictions are based solely on textual analysis of news articles "
                "and do not incorporate quantitative market data, technical indicators, "
                "or real-time pricing information. Pseudo-labels introduce noise. "
                "This tool is for research purposes and should not be used as the "
                "sole basis for financial decisions."
            ),
        }
