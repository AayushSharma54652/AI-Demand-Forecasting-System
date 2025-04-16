// JavaScript for chart generation and visualization

// Function to create forecast charts using Plotly.js
function createForecastChart(containerId, historicalData, forecastData, title = 'Demand Forecast') {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract data for plotting
    const historicalDates = historicalData.map(d => d.date || d.index);
    const historicalValues = historicalData.map(d => d.value || d[Object.keys(d)[0]]);
    
    const forecastDates = forecastData.map(d => d.date || d.index);
    const forecastValues = forecastData.map(d => d.forecast || d[Object.keys(d)[0]]);
    const lowerBounds = forecastData.map(d => d.lower_bound || d[Object.keys(d)[1]]);
    const upperBounds = forecastData.map(d => d.upper_bound || d[Object.keys(d)[2]]);
    
    // Create traces for Plotly
    const historicalTrace = {
        x: historicalDates,
        y: historicalValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Data',
        line: {
            color: 'blue',
            width: 2
        }
    };
    
    const forecastTrace = {
        x: forecastDates,
        y: forecastValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecast',
        line: {
            color: 'red',
            width: 2
        }
    };
    
    const upperBoundTrace = {
        x: forecastDates,
        y: upperBounds,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper Bound (95%)',
        line: {
            width: 0,
            color: 'rgba(255, 0, 0, 0.3)'
        },
        fill: 'tonexty',
        fillcolor: 'rgba(255, 0, 0, 0.1)'
    };
    
    const lowerBoundTrace = {
        x: forecastDates,
        y: lowerBounds,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound (95%)',
        line: {
            width: 0,
            color: 'rgba(255, 0, 0, 0.3)'
        }
    };
    
    const data = [historicalTrace, forecastTrace, lowerBoundTrace, upperBoundTrace];
    
    const layout = {
        title: title,
        xaxis: {
            title: 'Date',
            showgrid: true,
            zeroline: false
        },
        yaxis: {
            title: 'Demand',
            showgrid: true,
            zeroline: false
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        }
    };
    
    Plotly.newPlot(containerId, data, layout);
    
    return {
        updateView: function(view) {
            if (view === 'all') {
                Plotly.restyle(container, {'visible': true}, [0, 1, 2, 3]);
            } else if (view === 'forecast') {
                Plotly.restyle(container, {'visible': [false, true, true, true]}, [0, 1, 2, 3]);
            } else if (view === 'history') {
                Plotly.restyle(container, {'visible': [true, false, false, false]}, [0, 1, 2, 3]);
            }
        }
    };
}

// Function to create feature importance chart
function createFeatureImportanceChart(containerId, featureImportance) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract feature importance data
    const features = featureImportance.features;
    const importance = featureImportance.importance;
    
    // Sort features by importance
    const combinedData = features.map((feature, i) => ({ feature, importance: importance[i] }));
    combinedData.sort((a, b) => b.importance - a.importance);
    
    // Take top 10 or all if less than 10
    const topN = Math.min(10, combinedData.length);
    const topFeatures = combinedData.slice(0, topN);
    
    const sortedFeatures = topFeatures.map(d => d.feature);
    const sortedImportance = topFeatures.map(d => d.importance);
    
    const data = [{
        x: sortedImportance,
        y: sortedFeatures,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: 'rgba(50, 171, 96, 0.7)',
            line: {
                color: 'rgba(50, 171, 96, 1.0)',
                width: 1
            }
        }
    }];
    
    const layout = {
        title: 'Feature Importance',
        xaxis: {
            title: 'Importance Score',
            showgrid: true,
            zeroline: false
        },
        yaxis: {
            title: '',
            showgrid: true,
            zeroline: false
        },
        margin: {
            l: 150,
            r: 30,
            t: 50,
            b: 50
        }
    };
    
    Plotly.newPlot(containerId, data, layout);
}

// Function to create model comparison chart
function createModelComparisonChart(containerId, modelData) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract model data
    const models = modelData.map(m => m.name);
    const accuracy = modelData.map(m => m.accuracy);
    
    const data = [{
        x: models,
        y: accuracy,
        type: 'bar',
        marker: {
            color: 'rgba(0, 123, 255, 0.7)',
            line: {
                color: 'rgba(0, 123, 255, 1.0)',
                width: 1
            }
        }
    }];
    
    const layout = {
        title: 'Model Comparison',
        xaxis: {
            title: '',
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            title: 'Accuracy Score',
            showgrid: true,
            zeroline: false,
            range: [0, 1]
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 100
        }
    };
    
    Plotly.newPlot(containerId, data, layout);
}

// Function to create seasonal decomposition chart
function createSeasonalDecompositionChart(containerId, seasonalData) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract seasonal data
    const dates = seasonalData.dates;
    const observed = seasonalData.observed;
    const trend = seasonalData.trend;
    const seasonal = seasonalData.seasonal;
    const residual = seasonalData.residual;
    
    const traces = [
        {
            x: dates,
            y: observed,
            type: 'scatter',
            mode: 'lines',
            name: 'Observed',
            line: { color: 'blue' }
        },
        {
            x: dates,
            y: trend,
            type: 'scatter',
            mode: 'lines',
            name: 'Trend',
            line: { color: 'red' }
        },
        {
            x: dates,
            y: seasonal,
            type: 'scatter',
            mode: 'lines',
            name: 'Seasonal',
            line: { color: 'green' }
        },
        {
            x: dates,
            y: residual,
            type: 'scatter',
            mode: 'lines',
            name: 'Residual',
            line: { color: 'gray' }
        }
    ];
    
    const layout = {
        title: 'Seasonal Decomposition',
        xaxis: {
            title: 'Date',
            showgrid: true
        },
        yaxis: {
            title: 'Value',
            showgrid: true
        },
        hovermode: 'closest',
        legend: {
            orientation: 'h',
            y: -0.2
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 80
        }
    };
    
    Plotly.newPlot(containerId, traces, layout);
    
    return {
        updateView: function(component) {
            if (component === 'all') {
                Plotly.restyle(container, {'visible': true}, [0, 1, 2, 3]);
            } else if (component === 'observed') {
                Plotly.restyle(container, {'visible': [true, false, false, false]}, [0, 1, 2, 3]);
            } else if (component === 'trend') {
                Plotly.restyle(container, {'visible': [false, true, false, false]}, [0, 1, 2, 3]);
            } else if (component === 'seasonal') {
                Plotly.restyle(container, {'visible': [false, false, true, false]}, [0, 1, 2, 3]);
            } else if (component === 'residual') {
                Plotly.restyle(container, {'visible': [false, false, false, true]}, [0, 1, 2, 3]);
            }
        }
    };
}

// Function to create a heatmap for correlation matrix
function createCorrelationHeatmap(containerId, correlationData) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract correlation data
    const features = correlationData.features;
    const matrix = correlationData.matrix;
    
    const data = [{
        z: matrix,
        x: features,
        y: features,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1,
        showscale: true
    }];
    
    const layout = {
        title: 'Feature Correlation Matrix',
        xaxis: {
            title: '',
            ticks: '',
            side: 'bottom'
        },
        yaxis: {
            title: '',
            ticks: '',
            autorange: 'reversed'
        },
        margin: {
            l: 120,
            r: 50,
            t: 50,
            b: 120
        }
    };
    
    Plotly.newPlot(containerId, data, layout);
}

// Function to create interactive forecast exploration chart
function createInteractiveForecastChart(containerId, historicalData, forecastData, featureData) {
    const container = document.getElementById(containerId);
    
    if (!container) return;
    
    // Extract data
    const historicalDates = historicalData.map(d => d.date || d.index);
    const historicalValues = historicalData.map(d => d.value || d[Object.keys(d)[0]]);
    
    const forecastDates = forecastData.map(d => d.date || d.index);
    const forecastValues = forecastData.map(d => d.forecast || d[Object.keys(d)[0]]);
    
    // Create the main demand trace
    const demandTrace = {
        x: [...historicalDates, ...forecastDates],
        y: [...historicalValues, ...forecastValues],
        type: 'scatter',
        mode: 'lines',
        name: 'Demand',
        line: {
            color: 'blue',
            width: 2,
            dash: 'solid'
        }
    };
    
    // Create initial plot with just demand
    Plotly.newPlot(containerId, [demandTrace], {
        title: 'Interactive Forecast Exploration',
        xaxis: {
            title: 'Date',
            showgrid: true
        },
        yaxis: {
            title: 'Value',
            showgrid: true
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 80
        }
    });
    
    // Return an API for interacting with the chart
    return {
        // Add a feature line to the chart
        addFeature: function(featureName) {
            if (!featureData || !featureData[featureName]) return;
            
            const featureValues = featureData[featureName];
            
            // Normalize feature values to be in the same scale as demand
            const demandMin = Math.min(...historicalValues, ...forecastValues);
            const demandMax = Math.max(...historicalValues, ...forecastValues);
            const demandRange = demandMax - demandMin;
            
            const featureMin = Math.min(...featureValues);
            const featureMax = Math.max(...featureValues);
            const featureRange = featureMax - featureMin;
            
            const normalizedFeatureValues = featureValues.map(v => 
                ((v - featureMin) / featureRange) * demandRange + demandMin
            );
            
            const featureTrace = {
                x: [...historicalDates, ...forecastDates],
                y: normalizedFeatureValues,
                type: 'scatter',
                mode: 'lines',
                name: featureName,
                line: {
                    dash: 'dash'
                }
            };
            
            Plotly.addTraces(container, featureTrace);
        },
        
        // Remove a feature line from the chart
        removeFeature: function(featureName) {
            const traces = container.data;
            const traceIndex = traces.findIndex(trace => trace.name === featureName);
            
            if (traceIndex > 0) {  // Don't remove the demand trace (index 0)
                Plotly.deleteTraces(container, traceIndex);
            }
        },
        
        // Clear all features except demand
        clearFeatures: function() {
            // Keep only the first trace (demand)
            const traces = container.data;
            if (traces.length > 1) {
                const indicesToRemove = Array.from(
                    { length: traces.length - 1 }, 
                    (_, i) => i + 1
                );
                Plotly.deleteTraces(container, indicesToRemove);
            }
        }
    };
}