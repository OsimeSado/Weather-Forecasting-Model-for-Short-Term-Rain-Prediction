(function () {
    "use strict";

    function readJSON(id) {
        var el = document.getElementById(id);
        if (!el) return null;
        try {
            return JSON.parse(el.textContent);
        } catch (err) {
            return null;
        }
    }

    var labels       = readJSON("forecastTimes") || [];
    var temperatures = (readJSON("forecastTemps") || []).map(Number);
    var humidities   = (readJSON("forecastHumidity") || []).map(Number);

    var canvas = document.getElementById("forecastChart");
    if (!canvas) return;

    var ctx = canvas.getContext("2d");

    var TEAL   = "#5eead4";
    var INDIGO = "#818cf8";
    var TEXT   = "#cbd5e1";
    var MUTED  = "#64748b";
    var GRID   = "rgba(255, 255, 255, 0.06)";

    function makeGradient(color) {
        var gradient = ctx.createLinearGradient(0, 0, 0, 280);
        gradient.addColorStop(0, color + "55");
        gradient.addColorStop(1, color + "00");
        return gradient;
    }

    var glowPlugin = {
        id: "pointGlow",
        beforeDatasetDraw: function (chart, args) {
            var meta = args.meta;
            if (!meta || !meta.data) return;
            chart.ctx.save();
            chart.ctx.shadowColor = meta.dataset.borderColor;
            chart.ctx.shadowBlur = 12;
        },
        afterDatasetDraw: function (chart) {
            chart.ctx.restore();
        }
    };

    var config = {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Temperature (°C)",
                    data: temperatures,
                    borderColor: TEAL,
                    backgroundColor: makeGradient(TEAL),
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: TEAL,
                    pointBorderColor: "#0f172a",
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    yAxisID: "yTemp"
                },
                {
                    label: "Humidity (%)",
                    data: humidities,
                    borderColor: INDIGO,
                    backgroundColor: makeGradient(INDIGO),
                    borderWidth: 2,
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: INDIGO,
                    pointBorderColor: "#0f172a",
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    yAxisID: "yHum"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
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
                        label: function (context) {
                            var dataset = context.dataset.label || "";
                            var value = context.parsed.y;
                            if (dataset.indexOf("Temperature") === 0) return " " + dataset + ": " + value + "°C";
                            if (dataset.indexOf("Humidity") === 0) return " " + dataset + ": " + value + "%";
                            return " " + dataset + ": " + value;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: GRID, drawBorder: false },
                    ticks: {
                        color: MUTED,
                        font: { family: "'Inter', sans-serif", size: 12 }
                    }
                },
                yTemp: {
                    type: "linear",
                    position: "left",
                    grid: { color: GRID, drawBorder: false },
                    ticks: {
                        color: TEAL,
                        font: { family: "'Inter', sans-serif", size: 11 },
                        callback: function (value) { return value + "°"; }
                    },
                    title: { display: false }
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
                        callback: function (value) { return value + "%"; }
                    }
                }
            },
            animation: {
                duration: 900,
                easing: "easeOutQuart"
            }
        },
        plugins: [glowPlugin]
    };

    if (typeof Chart === "undefined") {
        console.error("chartSetup: Chart.js not loaded.");
        return;
    }

    Chart.defaults.color = TEXT;
    Chart.defaults.borderColor = GRID;

    new Chart(ctx, config);

    // Animate the rain-probability ring
    var ring = document.querySelector(".ring-fill");
    if (ring) {
        var percent = parseFloat(ring.dataset.percent || "0");
        var circumference = 2 * Math.PI * 52;
        var offset = circumference * (1 - Math.min(Math.max(percent, 0), 100) / 100);

        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = circumference;

        if (percent >= 70) {
            ring.style.stroke = "#f87171";
        } else if (percent >= 40) {
            ring.style.stroke = "#fbbf24";
        } else {
            ring.style.stroke = "#5eead4";
        }

        requestAnimationFrame(function () {
            ring.style.strokeDashoffset = offset;
        });
    }
})();
