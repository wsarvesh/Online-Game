
Chart.defaults.global.defaultFontFamily = 'TrulyMadly';
Chart.defaults.global.defaultFontColor = '#858796';

function pie_graph(id,lbl,vals){
  var canv_id = "canvas_"+id;
  var ctx = document.getElementById(canv_id);
  var myPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: lbl,
      datasets: [{
        data: vals,
        backgroundColor: ['#646da3', '#008e8c','#007858'],
        hoverBackgroundColor: ['#484e73', '#006665','#006349'],
        hoverBorderColor: "rgba(234, 236, 244, 1)",
      }],
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#dddfeb',
        borderWidth: 3,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
            label: (tooltipItems, data) => {
              // var datasetLabel = data.datasets[tooltipItem.datasetIndex].label || '';
              var label_show = data.labels[tooltipItems.index];
              return label_show +' : '+ data.datasets[tooltipItems.datasetIndex].data[tooltipItems.index] + ' RECORDS';
            }
        }
      },
      legend: {
        display: false
      },
      cutoutPercentage: 5,
    },
  });
}

function doughnut_graph(id,lbl,vals){
  var canv_id = "canvas_"+id;
  var ctx = document.getElementById(canv_id);
  var col_num = Math.floor(vals.length/3) + 1;
  var bg = new Array(col_num).fill(['#646da3', '#008e8c','#007858']).flat();
  var hbg = new Array(col_num).fill(['#484e73', '#006665','#006349']).flat();
  var myPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: lbl,
      datasets: [{
        data: vals,
        backgroundColor: bg,
        hoverBackgroundColor: hbg,
        hoverBorderColor: "rgba(234, 236, 244, 1)",
      }],
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#dddfeb',
        borderWidth: 0.25,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
            label: (tooltipItems, data) => {
              // var datasetLabel = data.datasets[tooltipItem.datasetIndex].label || '';
              var label_show = data.labels[tooltipItems.index];
              return label_show +' : '+ data.datasets[tooltipItems.datasetIndex].data[tooltipItems.index] + ' RECORDS';
            }
        }
      },
      legend: {
        display: false
      },
      cutoutPercentage: 50,
    },
  });
}

function attribute_graph(id,lbl,vals){
  var canv_id = "canvas_"+id;
  var ctx = document.getElementById(canv_id);
  var col_num = Math.floor(vals.length/3) + 1;
  var bg = new Array(col_num).fill(['#646da3', '#008e8c','#007858']).flat();
  var hbg = new Array(col_num).fill(['#484e73', '#006665','#006349']).flat();
  var myPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: lbl,
      datasets: [{
        data: vals,
        backgroundColor: bg,
        hoverBackgroundColor: hbg,
        hoverBorderColor: "rgba(234, 236, 244, 1)",
      }],
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        borderColor: '#dddfeb',
        borderWidth: 3,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
            label: (tooltipItems, data) => {
              // var datasetLabel = data.datasets[tooltipItem.datasetIndex].label || '';
              var label_show = data.labels[tooltipItems.index];
              return label_show +' : '+ data.datasets[tooltipItems.datasetIndex].data[tooltipItems.index] + ' RECORDS';
            }
        }
      },
      legend: {
        display: false
      },
      cutoutPercentage: 5,
    },
  });
}

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

function bar_graph(id,labels,lbl1,vals1,minval,maxval){
  var canv_id = "canvas_"+id;
  var ctx = document.getElementById(canv_id);
  var bg1 = new Array(vals1.length).fill(['#646da3','#008e8c']).flat();
  var hbg1 = new Array(vals1.length).fill(['#484e73','#006665']).flat();
  var myBarChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: lbl1,
        backgroundColor: bg1,
        hoverBackgroundColor: hbg1,
        hoverBorderColor: "#ffffff",
        borderColor: "#ffffff",
        borderWidth: 2,
        data: vals1,
      }
      // {
      //   label: lbl2,
      //   backgroundColor: bg2,
      //   hoverBackgroundColor: hbg2,
      //   hoverBorderColor: "#ffffff",
      //   borderColor: "#ffffff",
      //   borderWidth: 2,
      //   data: vals2,
      // }
      ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 25,
          top: 25,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          ticks: {
         autoSkip: false
      },
          time: {
            unit: 'month'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 100
          },
          maxBarThickness: 25,
        }],
        yAxes: [{
          ticks: {
            min: minval,
            max: maxval,
            maxTicksLimit: 20,
            padding: 10,
            // Include a dollar sign in the ticks
            callback: function(value, index, values) {
              return number_format(value,2);
            }
          },
          gridLines: {
            color: "#014143",
            zeroLineColor: "#014143",
            drawBorder: false,
            // borderDash: [2],
            // zeroLineBorderDash: [2]
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
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + ':' + number_format(tooltipItem.yLabel,2);
          }
        }
      },
    }
  });
}
