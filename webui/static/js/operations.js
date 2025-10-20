/**
 * Operations Management
 * 
 * Handles triggering operations (separate, cleanup, MIDI, video) and monitoring job progress.
 */

// Active job streams
const activeStreams = new Map();

/**
 * Start separation operation
 */
async function startSeparate() {
    if (!currentProject) return;
    
    try {
        showLoading('Starting separation...');
        
        const result = await api.separate(currentProject.number, {
            device: 'cpu', // TODO: Make configurable in Phase 3
            wiener: null,
            eq: false
        });
        
        hideLoading();
        showToast('Separation started', 'success');
        monitorJob(result.job_id, 'separate');
        
    } catch (error) {
        hideLoading();
        console.error('Failed to start separation:', error);
        showToast(`Separation failed: ${error.message}`, 'error');
    }
}

/**
 * Start cleanup operation
 */
async function startCleanup() {
    if (!currentProject) return;
    
    try {
        showLoading('Starting cleanup...');
        
        const result = await api.cleanup(currentProject.number, {
            threshold_db: -30.0,
            ratio: 10.0,
            attack_ms: 1.0,
            release_ms: 100.0
        });
        
        hideLoading();
        showToast('Cleanup started', 'success');
        monitorJob(result.job_id, 'cleanup');
        
    } catch (error) {
        hideLoading();
        console.error('Failed to start cleanup:', error);
        showToast(`Cleanup failed: ${error.message}`, 'error');
    }
}

/**
 * Start MIDI conversion
 */
async function startMidi() {
    if (!currentProject) return;
    
    try {
        showLoading('Starting MIDI conversion...');
        
        const result = await api.stemsToMidi(currentProject.number, {
            onset_threshold: 0.3,
            onset_delta: 0.01,
            onset_wait: 3,
            hop_length: 512,
            min_velocity: 80,
            max_velocity: 110
        });
        
        hideLoading();
        showToast('MIDI conversion started', 'success');
        monitorJob(result.job_id, 'stems-to-midi');
        
    } catch (error) {
        hideLoading();
        console.error('Failed to start MIDI conversion:', error);
        showToast(`MIDI conversion failed: ${error.message}`, 'error');
    }
}

/**
 * Start video rendering
 */
async function startVideo() {
    if (!currentProject) return;
    
    try {
        showLoading('Starting video rendering...');
        
        const result = await api.renderVideo(currentProject.number, {
            fps: 60,
            width: 1920,
            height: 1080
        });
        
        hideLoading();
        showToast('Video rendering started', 'success');
        monitorJob(result.job_id, 'render-video');
        
    } catch (error) {
        hideLoading();
        console.error('Failed to start video rendering:', error);
        showToast(`Video rendering failed: ${error.message}`, 'error');
    }
}

/**
 * Monitor a job's progress via SSE
 */
function monitorJob(jobId, operationName) {
    // Show active jobs section
    document.getElementById('active-jobs-section').classList.remove('hidden');
    
    // Create job card
    addJobCard(jobId, operationName);
    
    // Connect to SSE stream
    const stream = api.streamJobStatus(jobId, {
        onUpdate: (job) => {
            updateJobCard(job);
            addConsoleLog(`[${job.operation}] Progress: ${job.progress}%`, 'info');
        },
        onComplete: (job) => {
            updateJobCard(job);
            addConsoleLog(`[${job.operation}] Completed successfully!`, 'success');
            showToast(`${capitalize(job.operation)} completed!`, 'success');
            
            // Clean up
            activeStreams.delete(jobId);
            
            // Reload project to update file lists
            setTimeout(() => {
                if (currentProject) {
                    selectProject(currentProject.number);
                }
            }, 1000);
        },
        onError: (job) => {
            updateJobCard(job);
            addConsoleLog(`[${job.operation}] Failed: ${job.error}`, 'error');
            showToast(`${capitalize(job.operation)} failed: ${job.error}`, 'error');
            
            // Clean up
            activeStreams.delete(jobId);
        },
        onConnectionError: (error) => {
            console.error('SSE connection error:', error);
            addConsoleLog(`Connection error for job ${jobId}`, 'error');
            activeStreams.delete(jobId);
        }
    });
    
    activeStreams.set(jobId, stream);
}

/**
 * Create a job card in the UI
 */
function addJobCard(jobId, operationName) {
    const container = document.getElementById('active-jobs-list');
    
    const icons = {
        'separate': 'fa-divide',
        'cleanup': 'fa-broom',
        'stems-to-midi': 'fa-music',
        'render-video': 'fa-video'
    };
    
    const card = document.createElement('div');
    card.id = `job-${jobId}`;
    card.className = 'bg-gray-800 rounded-lg p-4 border border-gray-700';
    card.innerHTML = `
        <div class="flex items-center justify-between mb-3">
            <div class="flex items-center">
                <i class="fas ${icons[operationName] || 'fa-cog'} text-larsnet-primary mr-2"></i>
                <span class="font-medium">${capitalize(operationName)}</span>
            </div>
            <button onclick="cancelJob('${jobId}')" class="text-gray-400 hover:text-red-500 transition-smooth">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="space-y-2">
            <div class="flex items-center justify-between text-sm">
                <span class="text-gray-400">Status: <span id="job-${jobId}-status" class="text-yellow-500">queued</span></span>
                <span id="job-${jobId}-progress" class="text-larsnet-primary font-semibold">0%</span>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                <div id="job-${jobId}-bar" class="bg-larsnet-primary h-full transition-smooth" style="width: 0%"></div>
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

/**
 * Update a job card with new data
 */
function updateJobCard(job) {
    const statusEl = document.getElementById(`job-${job.id}-status`);
    const progressEl = document.getElementById(`job-${job.id}-progress`);
    const barEl = document.getElementById(`job-${job.id}-bar`);
    
    if (!statusEl) return; // Card was removed
    
    // Update status
    statusEl.textContent = job.status;
    statusEl.className = {
        'queued': 'text-yellow-500',
        'running': 'text-blue-500',
        'completed': 'text-green-500',
        'failed': 'text-red-500',
        'cancelled': 'text-gray-500'
    }[job.status] || 'text-gray-500';
    
    // Update progress
    progressEl.textContent = `${job.progress}%`;
    barEl.style.width = `${job.progress}%`;
    
    // Change bar color based on status
    if (job.status === 'completed') {
        barEl.className = 'bg-green-500 h-full transition-smooth';
    } else if (job.status === 'failed') {
        barEl.className = 'bg-red-500 h-full transition-smooth';
    }
    
    // Remove card if job is done
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
        // Refresh current project to update status badges
        if (job.status === 'completed' && currentProject) {
            selectProject(currentProject.number);
        }
        
        setTimeout(() => {
            const card = document.getElementById(`job-${job.id}`);
            if (card) {
                card.style.opacity = '0';
                card.style.transform = 'translateX(100%)';
                setTimeout(() => card.remove(), 300);
            }
            
            // Hide section if no more jobs
            const container = document.getElementById('active-jobs-list');
            if (container.children.length === 0) {
                document.getElementById('active-jobs-section').classList.add('hidden');
            }
        }, 3000);
    }
}

/**
 * Cancel a job
 */
async function cancelJob(jobId) {
    try {
        await api.cancelJob(jobId);
        showToast('Job cancelled', 'info');
        
        // Close SSE stream
        const stream = activeStreams.get(jobId);
        if (stream) {
            stream.close();
            activeStreams.delete(jobId);
        }
        
        // Remove card
        const card = document.getElementById(`job-${jobId}`);
        if (card) {
            card.remove();
        }
        
    } catch (error) {
        console.error('Failed to cancel job:', error);
        showToast('Failed to cancel job', 'error');
    }
}

/**
 * Add a log entry to console
 */
function addConsoleLog(message, level = 'info') {
    const console = document.getElementById('console-output');
    const consoleSection = document.getElementById('console-section');
    
    // Show console if hidden
    if (consoleSection.classList.contains('hidden')) {
        consoleSection.classList.remove('hidden');
    }
    
    const colors = {
        'info': 'text-gray-400',
        'success': 'text-green-500',
        'error': 'text-red-500',
        'warning': 'text-yellow-500'
    };
    
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `${colors[level] || colors['info']} mb-1`;
    entry.textContent = `[${timestamp}] ${message}`;
    
    console.appendChild(entry);
    console.scrollTop = console.scrollHeight;
}

/**
 * Clear console output
 */
function clearConsole() {
    document.getElementById('console-output').innerHTML = 
        '<div class="text-gray-500">Console cleared...</div>';
}

/**
 * Capitalize first letter of string
 */
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).replace(/-/g, ' ');
}
