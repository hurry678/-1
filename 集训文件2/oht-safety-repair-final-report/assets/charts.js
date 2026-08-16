(function () {
  var s = getComputedStyle(document.documentElement);
  var accent = s.getPropertyValue("--accent").trim();
  var accent2 = s.getPropertyValue("--accent2").trim();
  var danger = s.getPropertyValue("--danger").trim();
  var ink = s.getPropertyValue("--ink").trim();
  var muted = s.getPropertyValue("--muted").trim();
  var rule = s.getPropertyValue("--rule").trim();

  function baseGrid() {
    return { left: 64, right: 24, top: 54, bottom: 48 };
  }

  var clearance = echarts.init(document.getElementById("chart-clearance"), null, { renderer: "svg" });
  clearance.setOption({
    animation: false,
    color: [danger, accent2],
    tooltip: { trigger: "axis", appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(3) + " mm"; } },
    legend: { top: 8, textStyle: { color: ink } },
    grid: baseGrid(),
    xAxis: { type: "category", data: ["问题 1", "问题 2", "问题 3"], axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
    yAxis: { type: "value", name: "净空 / mm", min: -650, max: 400, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: "修复前补充审计", type: "bar", data: [324.978, -363.000, -580.000], markLine: { symbol: "none", lineStyle: { color: accent, type: "dashed" }, data: [{ yAxis: 300, name: "硬门 300 mm" }] } },
      { name: "修复后保守下界", type: "bar", data: [324.977513, 324.977497, 324.977470] }
    ]
  });

  var transfer = echarts.init(document.getElementById("chart-transfer"), null, { renderer: "svg" });
  transfer.setOption({
    animation: false,
    color: [accent, accent2],
    tooltip: { trigger: "axis", appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(3) + " s"; } },
    legend: { top: 8, textStyle: { color: ink } },
    grid: baseGrid(),
    xAxis: { type: "category", data: ["问题 1", "问题 2", "问题 3"], axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
    yAxis: { type: "value", name: "AvgTransferTime / s", axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
    series: [
      { name: "修复前正式方案", type: "bar", data: [171.393750, 143.007358, 830.386420] },
      { name: "修复后正式方案", type: "bar", data: [171.393750, 142.946305, 854.551420] }
    ]
  });

  window.addEventListener("resize", function () {
    clearance.resize();
    transfer.resize();
  });
})();
