(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var el = document.getElementById('clearance-chart');
  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    color: [accent, accent2],
    tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(3) + ' mm'; } },
    legend: { top: 4, textStyle: { color: ink }, data: ['求解器内部摘要', '独立分段重建'] },
    grid: { left: 60, right: 24, top: 58, bottom: 52 },
    xAxis: {
      type: 'category',
      data: ['问题 1', '问题 2', '问题 3'],
      axisLabel: { color: muted },
      axisLine: { lineStyle: { color: rule } }
    },
    yAxis: {
      type: 'value',
      min: 298,
      max: 327,
      name: '最小净空 / mm',
      nameTextStyle: { color: muted },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule } }
    },
    series: [
      {
        name: '求解器内部摘要',
        type: 'bar',
        data: [325.0, 325.0, 325.0],
        itemStyle: { color: accent },
        label: { show: true, position: 'top', color: ink, formatter: '{c}' }
      },
      {
        name: '独立保守下界',
        type: 'bar',
        data: [324.977513, 324.977507, 324.977477],
        itemStyle: { color: accent2 },
        label: { show: true, position: 'top', color: ink, formatter: '{c}' },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: accent2, type: 'dashed', width: 2 },
          label: { color: accent2, formatter: '题目硬门 300 mm' },
          data: [{ yAxis: 300 }]
        }
      }
    ],
    backgroundColor: bg2
  });
  window.addEventListener('resize', function () { chart.resize(); });
})();
