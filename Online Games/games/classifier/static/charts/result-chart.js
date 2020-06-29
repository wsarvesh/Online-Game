Chart.defaults.global.defaultFontFamily = 'TrulyMadly';
Chart.defaults.global.defaultFontColor = '#858796';

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}

function acc_chart(x,p1,p2){
  var id = "PieChart" + x.toString();
  var ctx = document.getElementById(id);
  var myPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ["Accurate Testing","Missed Testing"],
      datasets: [{
        data: [p1,p2],
        backgroundColor: ['#646da3', '#008e8c'],
        hoverBackgroundColor: ['#484e73', '#006665'],
        hoverBorderColor: "rgba(234, 236, 244, 1)",
      }],
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
      },
      legend: {
        display: false
      },
      cutoutPercentage: 75,
    },
  });
}

function graph_chart(x,vals,name,lbl){
  var id = "Graph" + x.toString();
  var ctx = document.getElementById(id);
  var myBarChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: name,
      datasets: [{
        label: lbl,
        backgroundColor: ['#646da3', '#008e8c','#646da3', '#008e8c','#646da3'],
        hoverBackgroundColor: ['#484e73', '#006665','#484e73', '#006665','#484e73'],
        hoverBorderColor: "#ffffff",
        borderColor: "#ffffff",
        borderWidth: 2,
        data: vals,
      },
    ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          time: {
            unit: 'month'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 6
          },
          maxBarThickness: 25,
        }],
        yAxes: [{
          ticks: {
            min: 0,
            max: 100,
            maxTicksLimit: 5,
            padding: 10,
            // Include a dollar sign in the ticks
            callback: function(value, index, values) {
              return number_format(value,2) + "%";
            }
          },
          gridLines: {
            color: "#014143",
            zeroLineColor: "#014143",
            drawBorder: false,
          }
        }],
      },
      legend: {
        display: false
      },
      tooltips: {
        titleMarginBottom: 10,
        titleFontColor: '#6e707e',
        titleFontSize: 14,
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#ffffff',
        borderWidth: 1,
        xPadding: 5,
        yPadding: 5,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + ': ' + number_format(tooltipItem.yLabel,2)+ " %";
          }
        }
      },
    }
  });

}

function time_chart(x,vals1,vals2,name,lbl1,lbl2,maxt){
  var id = "Graph" + x.toString();
  var ctx = document.getElementById(id);
  var myBarChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: name,
      datasets: [{
        label: lbl1,
        backgroundColor: ['#646da3','#646da3','#646da3','#646da3','#646da3'],
        hoverBackgroundColor: ['#484e73','#484e73','#484e73','#484e73','#484e73'],
        hoverBorderColor: "#ffffff",
        borderColor: "#ffffff",
        borderWidth: 2,
        data: vals1,
      },
      {
        label: lbl2,
        backgroundColor: ['#008e8c','#008e8c','#008e8c','#008e8c','#008e8c'],
        hoverBackgroundColor: ['#006665','#006665','#006665','#006665','#006665'],
        hoverBorderColor: "#ffffff",
        borderColor: "#ffffff",
        borderWidth: 2,
        data: vals2,
      }
    ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          time: {
            unit: 'month'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 6
          },
          maxBarThickness: 25,
        }],
        yAxes: [{
          ticks: {
            min: 0,
            max: maxt,
            maxTicksLimit: 5,
            padding: 10,
            // Include a dollar sign in the ticks
            callback: function(value, index, values) {
              return number_format(value,2) + "ms";
            }
          },
          gridLines: {
            color: "#014143",
            zeroLineColor: "#014143",
            drawBorder: false,
          }
        }],
      },
      legend: {
        display: false
      },
      tooltips: {
        titleMarginBottom: 10,
        titleFontColor: '#6e707e',
        titleFontSize: 14,
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#ffffff',
        borderWidth: 1,
        xPadding: 5,
        yPadding: 5,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + ': ' + number_format(tooltipItem.yLabel,2)+ " ms";
          }
        }
      },
    }
  });

}