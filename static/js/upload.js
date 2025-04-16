// JavaScript for the file upload functionality

document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file');
    const submitBtn = document.getElementById('submit-btn');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    
    // Prevent default behavior for drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when dragging files over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('bg-light');
        dropArea.style.borderColor = '#0d6efd';
    }
    
    function unhighlight() {
        dropArea.classList.remove('bg-light');
        dropArea.style.borderColor = '#ccc';
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            updateFileInfo(files[0]);
        }
    }
    
    // Handle file selection via file input
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            updateFileInfo(this.files[0]);
        }
    });
    
    // Update file information when a file is selected
    function updateFileInfo(file) {
        // Validate file type
        const validExtensions = ['.csv', '.xlsx', '.xls'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(fileExtension)) {
            alert('Please upload a CSV or Excel file.');
            resetFileInput();
            return;
        }
        
        // Display file info
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.classList.remove('d-none');
        
        // Enable submit button
        submitBtn.disabled = false;
    }
    
    // Format file size for display
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Remove selected file
    removeFileBtn.addEventListener('click', function() {
        resetFileInput();
    });
    
    function resetFileInput() {
        fileInput.value = '';
        fileInfo.classList.add('d-none');
        submitBtn.disabled = true;
    }
    
    // Handle form submission
    const uploadForm = document.getElementById('upload-form');
    
    uploadForm.addEventListener('submit', function(e) {
        // Only for visual feedback - the actual form submission will proceed
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        submitBtn.disabled = true;
        
        // Show loading indicators
        const processingIndicator = document.createElement('div');
        processingIndicator.className = 'alert alert-info mt-4';
        processingIndicator.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm me-3" role="status"></div>
                <div>
                    <strong>AI Processing in Progress</strong>
                    <p class="mb-0">Analyzing your data and training AI models. This may take a few moments...</p>
                </div>
            </div>
        `;
        
        uploadForm.appendChild(processingIndicator);
    });
});