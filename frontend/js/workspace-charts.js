/* ============================================================
   Side-Step GUI — Chart Controls
   TensorBoard button, series toggles, axis toggle, smoothing,
   drag-to-zoom, shift+pan, wheel zoom, hover tooltip.
   Extracted from workspace.js for 400 LOC cap.
   ============================================================ */

const WorkspaceCharts = (() => {
  "use strict";

  /* ---- TensorBoard Button ---- */
  function initTensorBoardBtn() {
    $("btn-open-tensorboard")?.addEventListener("click", async () => {
      const outputDir = jkSourceText($("monitor-output-dir")).replace("Output: ", "").trim() || "";
      const logDir = $("full-log-dir")?.value || (outputDir ? outputDir + "/runs" : "");
      const port = 6006;
      try {
        const result = await API.startTensorBoard(logDir, port, outputDir);
        if (result.ok) {
          window.open(result.url || ("http://localhost:" + port), "_blank");
          const verb = result.restarted ? "TensorBoard restarted" : "TensorBoard started";
          if (typeof showToast === "function") showToast(verb + (result.log_dir ? " (" + result.log_dir + ")" : ""), "ok");
        } else {
          if (typeof showToast === "function") showToast("TensorBoard failed: " + (result.error || "unknown error"), "error");
        }
      } catch (e) {
        console.error('[TensorBoard] API call failed, opening fallback URL:', e);
        window.open("http://localhost:" + port, "_blank");
      }
    });
  }

  /* ---- Chart Controls (TensorBoard-style) ---- */
  function initChartControls() {
    const _setPressed = (btn, isPressed) => {
      if (!btn) return;
      btn.setAttribute("aria-pressed", isPressed ? "true" : "false");
    };

    document.querySelectorAll(".chart-toggle").forEach((btn) => {
      _setPressed(btn, btn.classList.contains("active"));
      btn.addEventListener("click", () => {
        btn.classList.toggle("active");
        const isActive = btn.classList.contains("active");
        _setPressed(btn, isActive);
        const series = btn.dataset.series;
        const lineMap = { loss: "monitor-loss-line", ma5: "monitor-ma-line", lr: "monitor-lr-line" };
        const lineEl = $(lineMap[series]);
        if (lineEl) lineEl.style.display = isActive ? "" : "none";
      });
    });

    const axisBtns = Array.from(document.querySelectorAll(".chart-axis"));
    const _setAxisActive = (activeBtn) => {
      axisBtns.forEach((b) => {
        const isActive = b === activeBtn;
        b.classList.toggle("active", isActive);
        _setPressed(b, isActive);
      });
    };

    axisBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        _setAxisActive(btn);
        if (typeof Training !== "undefined") {
          Training.setChartMode(btn.dataset.axis);
        }
      });
    });
    const initialAxis = axisBtns.find((b) => b.classList.contains("active")) || axisBtns[0];
    if (initialAxis) _setAxisActive(initialAxis);

    document.querySelectorAll(".chart-zoom").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.zoom === "reset" && typeof Training !== "undefined") Training.zoomReset();
      });
    });

    const smoothSlider = $("chart-smoothing");
    const smoothVal = $("chart-smoothing-val");
    if (smoothSlider) {
      smoothSlider.addEventListener("input", () => {
        const w = parseInt(smoothSlider.value, 10) / 1000;
        if (smoothVal) smoothVal.textContent = w.toFixed(2);
        if (typeof Training !== "undefined") Training.setSmoothing(w);
      });
    }

    // Snap-to-grid toggle (S button + keyboard shortcut)
    const snapBtn = $("btn-snap-grid");
    function _toggleSnap() {
      if (typeof Training === "undefined") return;
      const next = !Training.getSnap();
      Training.setSnap(next);
      if (snapBtn) {
        snapBtn.classList.toggle("active", next);
        _setPressed(snapBtn, next);
      }
    }
    if (snapBtn) {
      snapBtn.addEventListener("click", _toggleSnap);
    }
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input,textarea,select,[contenteditable]")) return;
      if (e.key === 's' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        _toggleSnap();
      }
      if (e.key === 'Escape' && typeof Training !== "undefined") {
        Training.collapseExpanded();
      }
    });

    // Close expanded chart button
    $("btn-close-expanded")?.addEventListener("click", () => {
      if (typeof Training !== "undefined") Training.collapseExpanded();
    });

    // Delegate to shared ChartInteraction utility (Training loaded later, callbacks are lazy)
    if (typeof ChartInteraction !== "undefined") {
      const T = () => typeof Training !== "undefined" ? Training : null;

      // Helper: linearly interpolate between two values
      function _lerp(arr, fractionalIdx) {
        if (!arr || arr.length === 0) return null;
        const lo = Math.floor(fractionalIdx);
        const hi = Math.ceil(fractionalIdx);
        if (lo < 0 || hi >= arr.length) return arr[Math.max(0, Math.min(arr.length - 1, Math.round(fractionalIdx)))];
        if (lo === hi) return arr[lo];
        const t = fractionalIdx - lo;
        return arr[lo] * (1 - t) + arr[hi] * t;
      }

      ChartInteraction.wire({
        areaId: "monitor-series-area", svgId: "monitor-loss-svg",
        getView: () => T()?.getChartView() || { startIdx: 0, endIdx: 1, totalLen: 1 },
        getSnap: () => T()?.getSnap() ?? true,
        setRange: (lo, hi) => T()?.setViewRange(lo, hi),
        zoomReset: () => T()?.zoomReset(),
        minRange: 2,
        zoomStep: 0.16,
        enableDoubleClickReset: true,
        formatTip: (idx) => {
          const t = T();
          const mode = t?.getChartMode?.() || "epoch";
          const label = mode === "step" ? "Step" : "Epoch";
          const intIdx = Math.round(idx);
          const data = t?.getDataAtIndex?.(intIdx);
          if (!data) return '<table><tr><th colspan="3">' + label + ' ' + intIdx.toLocaleString() + '</th></tr></table>';
          const rows = [];
          rows.push('<tr><th colspan="3">' + label + ' ' + intIdx.toLocaleString() + '</th></tr>');
          if (data.smoothed != null) {
            rows.push('<tr><td><span class="chart-tooltip__swatch" style="background:var(--primary)"></span></td><td>Loss</td><td>' + data.smoothed.toFixed(4) + '</td></tr>');
          }
          if (data.raw != null) {
            rows.push('<tr><td><span class="chart-tooltip__swatch" style="background:var(--primary);opacity:0.3"></span></td><td>Raw</td><td>' + data.raw.toFixed(4) + '</td></tr>');
          }
          if (data.ma5 != null) {
            rows.push('<tr><td><span class="chart-tooltip__swatch" style="background:var(--changed)"></span></td><td>MA5</td><td>' + data.ma5.toFixed(4) + '</td></tr>');
          }
          if (data.lr != null) {
            rows.push('<tr><td><span class="chart-tooltip__swatch" style="background:var(--secondary)"></span></td><td>LR</td><td>' + data.lr.toExponential(2) + '</td></tr>');
          }
          return '<table>' + rows.join('') + '</table>';
        },
        onHover: (idx) => {
          const t = T();
          const chartOpts = t?.getChartOpts?.();
          if (!chartOpts || typeof TrainingChart === "undefined") return;
          const intIdx = Math.round(idx);
          const coords = TrainingChart.getPointCoords(intIdx, chartOpts);
          const dotsG = $("monitor-hover-dots");
          if (!coords || !dotsG) { if (dotsG) dotsG.style.display = "none"; return; }
          dotsG.style.display = "";
          // In free mode, interpolate X position for smooth cursor following
          let x = coords.x;
          if (idx !== intIdx) {
            const { fullLoss, viewXMin, viewXMax } = chartOpts;
            const totalLen = fullLoss.length;
            const startIdx = viewXMin ?? 0, endIdx = viewXMax ?? totalLen;
            const visLen = endIdx - startIdx;
            const localFrac = (idx - startIdx) / Math.max(1, visLen - 1);
            x = localFrac * 1000; // vbW = 1000
          }
          // Convert viewBox coords (1000×500) to percentage for HTML overlay
          const vbW = 1000, vbH = 500;
          const xPct = (x / vbW * 100).toFixed(2) + "%";
          function _placeDot(el, yVb) {
            if (!el) return;
            el.style.left = xPct;
            el.style.top = (yVb / vbH * 100).toFixed(2) + "%";
          }
          _placeDot($("dot-smooth"), coords.smoothY);
          _placeDot($("dot-raw"), coords.lossY);
          _placeDot($("dot-ma5"), coords.ma5Y);
          _placeDot($("dot-lr"), coords.lrY);
        },
        onLeave: () => {
          const dotsG = $("monitor-hover-dots");
          if (dotsG) dotsG.style.display = "none";
        },
      });
    }
  }

  function init() {
    initTensorBoardBtn();
    initChartControls();
  }

  return { init };
})();
