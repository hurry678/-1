(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var el = document.getElementById('chart-clearance');
  var chart = echarts.init(el, null, { renderer: 'svg' });

  chart.setOption({
    animation: false,
    color: [accent],
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function (value) {
        return Number(value).toFixed(6) + ' mm';
      }
    },
    grid: { left: 70, right: 28, top: 35, bottom: 55 },
    xAxis: {
      type: 'category',
      data: ['问题1', '问题2', '问题3'],
      axisLabel: { color: muted },
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      min: 295,
      max: 330,
      name: '车身净空 / mm',
      nameTextStyle: { color: muted },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule } }
    },
    series: [{
      name: '保守下界',
      type: 'bar',
      barWidth: '42%',
      data: [324.977513, 324.977497, 324.977470],
      itemStyle: { color: accent, borderRadius: [5, 5, 0, 0] },
      label: {
        show: true,
        position: 'top',
        color: ink,
        formatter: function (params) {
          return Number(params.value).toFixed(3);
        }
      },
      markLine: {
        symbol: 'none',
        lineStyle: { color: accent2, width: 2, type: 'dashed' },
        label: {
          color: ink,
          backgroundColor: bg2,
          formatter: '300 mm 硬门'
        },
        data: [{ yAxis: 300 }]
      }
    }]
  });

  window.addEventListener('resize', function () {
    chart.resize();
  });
})();
