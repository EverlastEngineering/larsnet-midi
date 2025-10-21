/**
 * Main Application
 * 
 * Initializes the LarsNet Web UI and coordinates all modules.
 */

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Initialize the application
 */
async function initializeApp() {
    console.log('LarsNet Web UI initializing...');
    
    // Initialize settings manager
    window.settingsManager = new SettingsManager();
    
    // Check API health
    await checkHealth();
    
    // Load projects
    await loadProjects();
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup upload zone
    setupUploadZone();
    
    console.log('LarsNet Web UI ready');
    addConsoleLog('Application initialized', 'success');
}

/**
 * Check API health and update status indicator
 */
async function checkHealth() {
    const statusEl = document.getElementById('health-status');
    
    try {
        const healthy = await api.checkHealth();
        
        if (healthy) {
            statusEl.innerHTML = `
                <div class="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                <span class="text-sm text-gray-400">Connected</span>
            `;
        } else {
            throw new Error('API returned unhealthy status');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        statusEl.innerHTML = `
            <div class="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
            <span class="text-sm text-red-400">Disconnected</span>
        `;
        showToast('Cannot connect to API server', 'error');
    }
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Refresh projects button
    document.getElementById('refresh-projects-btn').addEventListener('click', loadProjects);
    
    // Operation buttons
    document.getElementById('btn-separate').addEventListener('click', startSeparate);
    document.getElementById('btn-cleanup').addEventListener('click', startCleanup);
    document.getElementById('btn-midi').addEventListener('click', startMidi);
    document.getElementById('btn-video').addEventListener('click', startVideo);
    
    // Console clear button
    document.getElementById('clear-console-btn').addEventListener('click', clearConsole);
    
    // File input
    document.getElementById('browse-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    
    // Sidebar file input
    document.getElementById('sidebar-browse-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    
    document.getElementById('file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

/**
 * Setup drag and drop for upload zone
 */
function setupUploadZone() {
    const uploadZone = document.getElementById('upload-section');
    const sidebarUploadZone = document.getElementById('sidebar-upload-section');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('drag-over');
        }, false);
    });
    
    // Handle dropped files
    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    }, false);
    
    // Setup sidebar upload zone drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        sidebarUploadZone.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        sidebarUploadZone.addEventListener(eventName, () => {
            sidebarUploadZone.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        sidebarUploadZone.addEventListener(eventName, () => {
            sidebarUploadZone.classList.remove('drag-over');
        }, false);
    });
    
    sidebarUploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    }, false);
}

/**
 * Prevent default drag/drop behavior
 */
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

/**
 * Handle file upload
 */
async function handleFileUpload(file) {
    console.log('Uploading file:', file.name);
    
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-aiff', 'audio/aiff'];
    const allowedExtensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif'];
    
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExt)) {
        showToast('Invalid file type. Please upload WAV, MP3, FLAC, or AIFF files.', 'error');
        return;
    }
    
    // Validate file size (500MB max)
    if (file.size > 500 * 1024 * 1024) {
        showToast('File is too large. Maximum size is 500MB.', 'error');
        return;
    }
    
    // Show upload progress
    const uploadZone = document.getElementById('upload-zone');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressPercent = document.getElementById('upload-percent');
    
    uploadZone.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    
    try {
        const result = await api.upload(file, (percent) => {
            progressBar.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
        });
        
        // Upload complete
        showToast('File uploaded successfully!', 'success');
        addConsoleLog(`Uploaded: ${file.name}`, 'success');
        
        // Reload projects
        await loadProjects();
        
        // Select the new project
        if (result.project) {
            await selectProject(result.project.number);
        }
        
    } catch (error) {
        console.error('Upload failed:', error);
        showToast(`Upload failed: ${error.message}`, 'error');
        addConsoleLog(`Upload failed: ${error.message}`, 'error');
    } finally {
        // Reset upload zone
        uploadZone.classList.remove('hidden');
        uploadProgress.classList.add('hidden');
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        
        // Clear file input
        document.getElementById('file-input').value = '';
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };
    
    const colors = {
        'success': 'bg-green-600',
        'error': 'bg-red-600',
        'warning': 'bg-yellow-600',
        'info': 'bg-blue-600'
    };
    
    const toast = document.createElement('div');
    toast.className = `${colors[type]} text-white px-6 py-4 rounded-lg shadow-lg flex items-center space-x-3 animate-slide-in max-w-md`;
    toast.innerHTML = `
        <i class="fas ${icons[type]} text-xl"></i>
        <span class="flex-1">${escapeHtml(message)}</span>
        <button onclick="this.parentElement.remove()" class="text-white hover:text-gray-200">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = document.getElementById('loading-message');
    messageEl.textContent = message;
    overlay.classList.remove('hidden');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

/**
 * Create job card HTML
 */
function createJobCard(job) {
    const icons = {
        'separate': 'fa-divide',
        'cleanup': 'fa-broom',
        'stems-to-midi': 'fa-music',
        'render-video': 'fa-video'
    };
    
    const statusColors = {
        'queued': 'text-yellow-500',
        'running': 'text-blue-500',
        'completed': 'text-green-500',
        'failed': 'text-red-500',
        'cancelled': 'text-gray-500'
    };
    
    return `
        <div id="job-${job.id}" class="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div class="flex items-center justify-between mb-3">
                <div class="flex items-center">
                    <i class="fas ${icons[job.operation] || 'fa-cog'} text-larsnet-primary mr-2"></i>
                    <span class="font-medium">${capitalize(job.operation)}</span>
                </div>
                <button onclick="cancelJob('${job.id}')" class="text-gray-400 hover:text-red-500 transition-smooth">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="space-y-2">
                <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-400">Status: <span id="job-${job.id}-status" class="${statusColors[job.status]}">${job.status}</span></span>
                    <span id="job-${job.id}-progress" class="text-larsnet-primary font-semibold">${job.progress}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                    <div id="job-${job.id}-bar" class="bg-larsnet-primary h-full transition-smooth" style="width: ${job.progress}%"></div>
                </div>
            </div>
        </div>
    `;
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slide-in {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .animate-slide-in {
        animation: slide-in 0.3s ease-out;
    }
`;
document.head.appendChild(style);
