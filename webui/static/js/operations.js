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
        // Get settings from SettingsManager
        const settings = window.settingsManager.getSettingsForOperation('separate');
        
        const result = await api.separate(currentProject.number, {
            device: settings.device,
            wiener: settings.wiener_exponent,
            eq: settings.apply_eq
        });
        
        showToast('Separation started', 'success');
        monitorJob(result.job_id, 'separate');
        
    } catch (error) {
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
        // Get settings from SettingsManager
        const settings = window.settingsManager.getSettingsForOperation('cleanup');
        
        const result = await api.cleanup(currentProject.number, {
            threshold_db: settings.threshold,
            ratio: settings.ratio,
            attack_ms: settings.attack,
            release_ms: settings.release
        });
        
        showToast('Cleanup started', 'success');
        monitorJob(result.job_id, 'cleanup');
        
    } catch (error) {
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
        // Get settings from SettingsManager
        const settings = window.settingsManager.getSettingsForOperation('midi');
        
        const result = await api.stemsToMidi(currentProject.number, {
            onset_threshold: settings.onset_threshold,
            onset_delta: settings.onset_delta,
            onset_wait: null, // Not exposed in basic UI
            hop_length: null, // Not exposed in basic UI
            min_velocity: settings.min_velocity,
            max_velocity: settings.max_velocity,
            tempo: settings.tempo,
            detect_hihat_open: settings.detect_hihat_open
        });
        
        showToast('MIDI conversion started', 'success');
        monitorJob(result.job_id, 'stems-to-midi');
        
    } catch (error) {
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
        // Get settings from SettingsManager
        const settings = window.settingsManager.getSettingsForOperation('video');
        
        // Parse resolution
        let width = 1920, height = 1080;
        if (settings.resolution === '1440p') {
            width = 2560; height = 1440;
        } else if (settings.resolution === '4k') {
            width = 3840; height = 2160;
        }
        
        const result = await api.renderVideo(currentProject.number, {
            fps: parseInt(settings.fps),
            width: width,
            height: height,
            include_audio: settings.includeAudio || false
        });
        
        showToast('Video rendering started', 'success');
        monitorJob(result.job_id, 'render-video');
        
    } catch (error) {
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
            
            // Display new logs from the update (SSE now sends only new_logs, not full logs array)
            if (job.new_logs && job.new_logs.length > 0) {
                job.new_logs.forEach(log => {
                    addConsoleLog(`[${operationName}] ${log.message}`, log.level);
                });
            }
        },
        onComplete: (job) => {
            updateJobCard(job);
            addConsoleLog(`[${job.operation}] Completed successfully!`, 'success');
            showToast(`${capitalize(job.operation)} completed!`, 'success');
            
            // Clean up
            activeStreams.delete(jobId);
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
    card.className = 'bg-gray-800 rounded-lg p-4 border border-gray-700 transition-smooth';
    card.innerHTML = `
        <div class="flex items-center justify-between mb-3">
            <div class="flex items-center">
                <i class="fas ${icons[operationName] || 'fa-cog'} text-larsnet-primary mr-2"></i>
                <span class="font-medium">${capitalize(operationName)}</span>
            </div>
            <button id="job-${jobId}-close-btn" data-job-id="${jobId}" class="text-gray-400 hover:text-red-500 transition-smooth">
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
    
    // Start collapsed for animation
    card.style.height = '0';
    card.style.opacity = '0';
    card.style.overflow = 'hidden';
    card.style.marginBottom = '0';
    card.style.paddingTop = '0';
    card.style.paddingBottom = '0';
    
    container.appendChild(card);
    
    // Get the natural height before animating
    card.style.height = 'auto';
    const height = card.offsetHeight;
    card.style.height = '0';
    
    // Force reflow to ensure starting state is applied
    card.offsetHeight;
    
    // Use requestAnimationFrame to ensure smooth animation
    requestAnimationFrame(() => {
        card.style.height = height + 'px';
        card.style.opacity = '1';
        card.style.marginBottom = '0.75rem';
        card.style.paddingTop = '1rem';
        card.style.paddingBottom = '1rem';
        
        // Remove inline height after animation so it can adapt to content
        setTimeout(() => {
            card.style.height = 'auto';
        }, 300);
    });
    
    // Attach event listener for close/cancel button
    const closeBtn = document.getElementById(`job-${jobId}-close-btn`);
    if (closeBtn) {
        closeBtn.addEventListener('click', () => cancelJob(jobId));
    }
}

/**
 * Update a job card with new data
 */
function updateJobCard(job) {
    const statusEl = document.getElementById(`job-${job.id}-status`);
    const progressEl = document.getElementById(`job-${job.id}-progress`);
    const barEl = document.getElementById(`job-${job.id}-bar`);
    
    if (!statusEl) return; // Card was removed
    
    // Update status - show detailed status if available, otherwise use basic status
    const displayStatus = job.status_detail || job.status;
    statusEl.textContent = displayStatus;
    const statusLower = job.status.toLowerCase();
    statusEl.className = {
        'queued': 'text-yellow-500',
        'running': 'text-blue-500',
        'completed': 'text-green-500',
        'failed': 'text-red-500',
        'cancelled': 'text-gray-500'
    }[statusLower] || 'text-gray-500';
    
    // Update progress
    progressEl.textContent = `${job.progress}%`;
    barEl.style.width = `${job.progress}%`;
    
    // Change bar color based on status
    if (statusLower === 'completed') {
        barEl.className = 'bg-green-500 h-full transition-smooth';
    } else if (statusLower === 'failed') {
        barEl.className = 'bg-red-500 h-full transition-smooth';
    }
    
    // Remove card if job is done
    if (statusLower === 'completed' || statusLower === 'failed' || statusLower === 'cancelled') {
        // Change X button to close instead of cancel
        const closeBtn = document.getElementById(`job-${job.id}-close-btn`);
        if (closeBtn) {
            // Remove inline onclick attribute
            closeBtn.removeAttribute('onclick');
            // Set up new click handler
            closeBtn.onclick = (e) => {
                e.preventDefault();
                removeJobCard(job.id);
            };
            closeBtn.className = 'text-gray-400 hover:text-gray-200 transition-smooth';
        }
        
        // Refresh project data to update status badges
        if (statusLower === 'completed') {
            // Refresh project list (left sidebar badges)
            loadProjects();
            
            // Refresh current project files without reloading jobs
            if (currentProject) {
                refreshCurrentProjectFiles();
            }
        }
        
        // Auto-remove after 2 seconds
        setTimeout(() => {
            removeJobCard(job.id);
        }, 2000);
    }
}

/**
 * Refresh current project's files without reloading jobs
 */
async function refreshCurrentProjectFiles() {
    if (!currentProject) return;
    
    try {
        const data = await api.getProject(currentProject.number);
        currentProject = data.project;
        
        // Update UI elements that depend on files
        updateProjectHeader();
        updateOperationButtons();
        updateDownloads();
        renderProjectsList(); // Update sidebar badges
    } catch (error) {
        console.error('Failed to refresh project files:', error);
    }
}

/**
 * Remove a job card from UI
 */
function removeJobCard(jobId) {
    const card = document.getElementById(`job-${jobId}`);
    if (card) {
        // First fade out and shrink
        card.style.opacity = '0';
        card.style.transform = 'translateY(-20px)';
        
        // Then collapse height smoothly
        setTimeout(() => {
            const height = card.offsetHeight;
            card.style.height = height + 'px';
            card.style.overflow = 'hidden';
            card.style.marginBottom = '0.75rem';
            
            // Force reflow
            card.offsetHeight;
            
            // Animate to zero height
            card.style.height = '0';
            card.style.marginBottom = '0';
            card.style.paddingTop = '0';
            card.style.paddingBottom = '0';
            
            // Remove after animation
            setTimeout(() => {
                card.remove();
                
                // Hide section if no more jobs
                const container = document.getElementById('active-jobs-list');
                if (container.children.length === 0) {
                    document.getElementById('active-jobs-section').classList.add('hidden');
                }
            }, 300);
        }, 300);
    }
}

/**
 * Cancel a job (or just remove the card if already completed)
 */
async function cancelJob(jobId) {
    const card = document.getElementById(`job-${jobId}`);
    
    // Check if job is already completed by looking at status text color
    const statusEl = document.getElementById(`job-${jobId}-status`);
    if (statusEl && (statusEl.classList.contains('text-green-500') || statusEl.classList.contains('text-red-500') || statusEl.classList.contains('text-gray-500'))) {
        // Job is already done, just remove the card
        removeJobCard(jobId);
        return;
    }
    
    // Job is still running, try to cancel it
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
        if (card) {
            removeJobCard(jobId);
        }
        
    } catch (error) {
        console.error('Failed to cancel job:', error);
        // If cancel fails but the card exists, just remove it anyway
        if (card) {
            removeJobCard(jobId);
        }
    }
}

/**
 * Add a log entry to console
 */
function addConsoleLog(message, level = 'info') {
    const console = document.getElementById('console-output');
    const consoleSection = document.getElementById('console-section');
    
    // Show console section if hidden (but keep it collapsed)
    if (consoleSection.classList.contains('hidden')) {
        consoleSection.classList.remove('hidden');
    }
    
    // Update log count badge
    updateConsoleLogCount();
    
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
    updateConsoleLogCount();
}

/**
 * Update console log count badge
 */
function updateConsoleLogCount() {
    const console = document.getElementById('console-output');
    const badge = document.getElementById('console-badge');
    const logCount = console.children.length;
    badge.textContent = `${logCount} log${logCount !== 1 ? 's' : ''}`;
}

/**
 * Toggle console expanded/collapsed
 */
function toggleConsole() {
    const container = document.getElementById('console-output-container');
    const icon = document.getElementById('console-toggle-icon');
    const isCollapsed = container.style.maxHeight === '0px' || container.style.maxHeight === '';
    
    if (isCollapsed) {
        // Expand
        container.style.maxHeight = '300px';
        icon.classList.remove('fa-chevron-right');
        icon.classList.add('fa-chevron-down');
    } else {
        // Collapse
        container.style.maxHeight = '0';
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-right');
    }
}

/**
 * Capitalize first letter of string
 */
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).replace(/-/g, ' ');
}
