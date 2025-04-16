// JavaScript for the main dashboard functionality

document.addEventListener('DOMContentLoaded', function() {
    // Animated counters for statistics (if they exist)
    const counterElements = document.querySelectorAll('.counter-value');
    
    if (counterElements.length > 0) {
        counterElements.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'));
            const duration = 1500; // milliseconds
            const step = Math.ceil(target / (duration / 16)); // 60fps
            
            let current = 0;
            const timer = setInterval(() => {
                current += step;
                counter.textContent = current;
                
                if (current >= target) {
                    counter.textContent = target;
                    clearInterval(timer);
                }
            }, 16);
        });
    }
    
    // Model info API call for the dashboard (if the element exists)
    const modelInfoContainer = document.getElementById('model-info');
    
    if (modelInfoContainer) {
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                // Create model cards
                let modelCardsHtml = '';
                
                data.forEach(model => {
                    const accuracyClass = model.accuracy >= 0.9 ? 'text-success' :
                                        model.accuracy >= 0.8 ? 'text-primary' : 'text-warning';
                    
                    modelCardsHtml += `
                        <div class="col-md-4 mb-4">
                            <div class="card h-100">
                                <div class="card-body">
                                    <h5 class="card-title">${model.name}</h5>
                                    <div class="d-flex justify-content-between mb-3">
                                        <span>Accuracy:</span>
                                        <span class="fw-bold ${accuracyClass}">${(model.accuracy * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <span>Latency:</span>
                                        <span>${model.latency}</span>
                                    </div>
                                </div>
                                <div class="card-footer bg-transparent">
                                    <button class="btn btn-sm btn-outline-primary learn-more-btn" 
                                            data-model="${model.name}">
                                        Learn more
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                modelInfoContainer.innerHTML = modelCardsHtml;
                
                // Add event listeners to the learn more buttons
                document.querySelectorAll('.learn-more-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const modelName = this.getAttribute('data-model');
                        showModelInfo(modelName);
                    });
                });
            })
            .catch(error => {
                console.error('Error fetching model data:', error);
                modelInfoContainer.innerHTML = `
                    <div class="col-12">
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Unable to load model information. Please try again later.
                        </div>
                    </div>
                `;
            });
    }
    
    function showModelInfo(modelName) {
        // This would typically show a modal with model details
        // For simplicity, we'll just use an alert
        const modelDescriptions = {
            'ARIMA': 'AutoRegressive Integrated Moving Average (ARIMA) is a statistical model for analyzing and forecasting time series data. It combines autoregression, differencing, and moving average components.',
            'Prophet': 'Prophet is a forecasting procedure developed by Facebook that works best with time series that have strong seasonal effects and several seasons of historical data.',
            'LSTM': 'Long Short-Term Memory (LSTM) is a type of recurrent neural network capable of learning long-term dependencies, particularly useful for time series prediction.',
            'Linear Regression': 'Linear Regression is a statistical model that analyzes the relationship between a dependent variable and one or more independent variables by fitting a linear equation.',
            'Ridge': 'Ridge Regression is a regularization technique for analyzing multiple regression data that suffer from multicollinearity.',
            'Random Forest': 'Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.',
            'Gradient Boosting': 'Gradient Boosting is a machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.'
        };
        
        const description = modelDescriptions[modelName] || 'No detailed information available for this model.';
        
        // Create a Bootstrap modal dynamically
        const modalHtml = `
            <div class="modal fade" id="modelInfoModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${modelName} Model</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>${description}</p>
                            
                            <h6 class="mt-4">Use Cases:</h6>
                            <ul>
                                <li>Time series forecasting</li>
                                <li>Demand prediction</li>
                                <li>Sales forecasting</li>
                                <li>Inventory optimization</li>
                            </ul>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Append modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Initialize and show the modal
        const modalElement = document.getElementById('modelInfoModal');
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
        
        // Remove modal from DOM after it's hidden
        modalElement.addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }
    
    // Handle any feature demos on the dashboard
    const demoButtons = document.querySelectorAll('.demo-btn');
    
    if (demoButtons.length > 0) {
        demoButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const demoType = this.getAttribute('data-demo');
                
                if (demoType === 'quick-forecast') {
                    showQuickForecastDemo();
                } else if (demoType === 'feature-importance') {
                    showFeatureImportanceDemo();
                }
            });
        });
    }
    
    function showQuickForecastDemo() {
        // Here you would implement a quick forecast demo
        alert('Quick forecast demo would be shown here.');
    }
    
    function showFeatureImportanceDemo() {
        // Here you would implement a feature importance demo
        alert('Feature importance demo would be shown here.');
    }
});