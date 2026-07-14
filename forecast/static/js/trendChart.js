(function () {
    "use strict";

    var timesEl = document.getElementById("trendTimes");
    var tempsEl = document.getElementById("trendTemps");
    var humsEl = document.getElementById("trendHumidities");
    var posEl = document.getElementById("trendCurrentPos");

    if (!timesEl || !tempsEl || !humsEl) return;

    var times = JSON.parse(timesEl.textContent);
    var temps = JSON.parse(tempsEl.textContent);
    var humidities = JSON.parse(humsEl.textContent);
    var currentPos = posEl ? JSON.parse(posEl.textContent) : temps.length - 1;

    var canvas = document.getElementById("trendChart");
    if (!canvas || typeof Chart === "undefined") return;

    var ctx = canvas.getContext("2d");

    var TEAL = "#5eead4";
    var INDIGO = "#818cf8";
    var AMBER = "#fbbf24";
    var TEXT = "#cbd5e1";
    var MUTED = "#64748b";
    var GRID = "rgba(255, 255, 255, 0.06)";

    function makeGradient(color) {
        var gradient = ctx.createLinearGradient(0, 0, 0, 320);
        gradient.addColorStop(0, color + "44");
        gradient.addColorStop(1, color + "00");
        return gradient;
    }

    var pointColors = temps.map(function (_, i) {
        return i === currentPos ? AMBER : TEAL;
    });
    var pointRadii = temps.map(function (_, i) {
        return i === currentPos ? 7 : 2;
    });
    var pointBorders = temps.map(function (_, i) {
        return i === currentPos ? 3 : 1;
    });

    var humPointColors = humidities.map(function (_, i) {
        return i === currentPos ? AMBER : INDIGO;
    });
    var humPointRadii = humidities.map(function (_, i) {
        return i === currentPos ? 6 : 2;
    });

    var nowLine = {
        id: "nowLine",
        afterDatasetsDraw: function (chart) {
            if (currentPos < 0 || currentPos >= chart.data.labels.length) return;
            var xScale = chart.scales.x;
            var x = xScale.getPixelForValue(currentPos);
            var yTop = chart.chartArea.top;
            var yBottom = chart.chartArea.bottom;

            chart.ctx.save();
            chart.ctx.beginPath();
            chart.ctx.setLineDash([4, 4]);
            chart.ctx.strokeStyle = AMBER;
            chart.ctx.lineWidth = 1.5;
            chart.ctx.moveTo(x, yTop);
            chart.ctx.lineTo(x, yBottom);
            chart.ctx.stroke();

            chart.ctx.fillStyle = AMBER;
            chart.ctx.font = "600 11px 'Inter', sans-serif";
            chart.ctx.textAlign = "center";
            chart.ctx.fillText("Now", x, yTop - 6);
            chart.ctx.restore();
        }
    };

    var showEvery = Math.max(1, Math.floor(times.length / 12));
    var tickLabels = times.map(function (t, i) {
        if (i === currentPos) return t;
        if (i % showEvery === 0) return t;
        return "";
    });

    new Chart(ctx, {
        type: "line",
        data: {
            labels: tickLabels,
            datasets: [
                {
                    label: "Temperature (°C)",
                    data: temps,
                    borderColor: TEAL,
                    backgroundColor: makeGradient(TEAL),
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: "#0f172a",
                    pointBorderWidth: pointBorders,
                    pointRadius: pointRadii,
                    pointHoverRadius: 6,
                    yAxisID: "yTemp"
                },
                {
                    label: "Humidity (%)",
                    data: humidities,
                    borderColor: INDIGO,
                    backgroundColor: "transparent",
                    borderWidth: 2,
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0.35,
                    pointBackgroundColor: humPointColors,
                    pointBorderColor: "#0f172a",
                    pointBorderWidth: 1,
                    pointRadius: humPointRadii,
                    pointHoverRadius: 5,
                    yAxisID: "yHum"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: {
                    position: "top",
                    align: "end",
                    labels: {
                        color: TEXT,
                        font: { family: "'Inter', sans-serif", size: 12, weight: "500" },
                        usePointStyle: true,
                        pointStyle: "circle",
                        padding: 16,
                        boxWidth: 8,
                        boxHeight: 8
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.95)",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    titleColor: "#f8fafc",
                    titleFont: { family: "'Inter', sans-serif", size: 13, weight: "600" },
                    bodyColor: TEXT,
                    bodyFont: { family: "'Inter', sans-serif", size: 12 },
                    padding: 12,
                    cornerRadius: 10,
                    displayColors: true,
                    boxPadding: 6,
                    callbacks: {
                        title: function (items) {
                            var idx = items[0].dataIndex;
                            return times[idx];
                        },
                        label: function (context) {
                            var ds = context.dataset.label || "";
                            var v = context.parsed.y;
                            if (ds.indexOf("Temperature") === 0) return " " + ds + ": " + v + "°C";
                            if (ds.indexOf("Humidity") === 0) return " " + ds + ": " + v + "%";
                            return " " + ds + ": " + v;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: GRID, drawBorder: false },
                    ticks: {
                        color: MUTED,
                        font: { family: "'Inter', sans-serif", size: 11 },
                        maxRotation: 0,
                        autoSkip: false
                    }
                },
                yTemp: {
                    type: "linear",
                    position: "left",
                    grid: { color: GRID, drawBorder: false },
                    ticks: {
                        color: TEAL,
                        font: { family: "'Inter', sans-serif", size: 11 },
                        callback: function (v) { return v + "°"; }
                    }
                },
                yHum: {
                    type: "linear",
                    position: "right",
                    min: 0,
                    max: 100,
                    grid: { drawOnChartArea: false, drawBorder: false },
                    ticks: {
                        color: INDIGO,
                        font: { family: "'Inter', sans-serif", size: 11 },
                        callback: function (v) { return v + "%"; }
                    }
                }
            },
            animation: { duration: 900, easing: "easeOutQuart" }
        },
        plugins: [nowLine]
    });
})();
