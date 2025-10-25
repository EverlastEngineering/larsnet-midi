/**
 * Projects Management
 * 
 * Handles project list rendering, selection, and status display.
 */

let currentProject = null;
let allProjects = [];

/**
 * Load and display all projects
 */
async function loadProjects() {
    try {
        const data = await api.getProjects();
        allProjects = data.projects;
        renderProjectsList();
    } catch (error) {
        console.error('Failed to load projects:', error);
        showToast('Failed to load projects', 'error');
        document.getElementById('projects-list').innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                <p class="text-sm">Failed to load projects</p>
                <button onclick="loadProjects()" class="mt-2 text-larsnet-primary hover:underline text-xs">
                    Retry
                </button>
            </div>
        `;
    }
}

/**
 * Render projects list in sidebar
 */
function renderProjectsList() {
    const container = document.getElementById('projects-list');
    
    if (allProjects.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-folder-open text-3xl mb-2"></i>
                <p class="text-sm">No projects yet</p>
                <p class="text-xs mt-1">Upload an audio file to get started</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = allProjects.map(project => `
        <div class="project-item p-3 rounded-lg border border-gray-700 hover:border-gray-600 transition-smooth cursor-pointer mb-2 ${currentProject && currentProject.number === project.number ? 'bg-larsnet-primary bg-opacity-10 border-larsnet-primary' : 'bg-gray-800'}"
             onclick="selectProject(${project.number})">
            <div class="flex items-start justify-between">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center">
                        <span class="text-xs font-semibold text-gray-500 mr-2">#${project.number}</span>
                        <h3 class="font-medium text-sm truncate">${escapeHtml(project.name)}</h3>
                    </div>
                    <div class="flex items-center mt-2 space-x-2 text-xs">
                        ${getStatusBadge('stems', project.has_stems)}
                        ${getStatusBadge('cleaned', project.has_cleaned)}
                        ${getStatusBadge('midi', project.has_midi)}
                        ${getStatusBadge('video', project.has_video)}
                    </div>
                </div>
                ${currentProject && currentProject.number === project.number ? '<i class="fas fa-check-circle text-larsnet-primary ml-2"></i>' : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Get status badge HTML
 */
function getStatusBadge(type, hasIt) {
    const icons = {
        stems: 'fa-divide',
        cleaned: 'fa-broom',
        midi: 'fa-music',
        video: 'fa-video'
    };
    
    const colors = hasIt ? 'text-green-500' : 'text-gray-600';
    
    return `<i class="fas ${icons[type]} ${colors}" title="${type}"></i>`;
}

/**
 * Select a project and load its details
 */
async function selectProject(projectNumber) {
    try {
        const data = await api.getProject(projectNumber);
        currentProject = data.project;
        window.currentProject = currentProject; // Expose to window for other modules
        
        renderProjectsList(); // Re-render to update active state
        updateProjectHeader();
        updateOperationButtons();
        updateDownloads();
        
        // Show project section, hide landing and main upload
        document.getElementById('landing-section').classList.add('hidden');
        document.getElementById('project-section').classList.remove('hidden');
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('console-section').classList.remove('hidden');
        
        // Close mobile sidebar after selecting project
        if (window.innerWidth < 768) {
            const projectsSidebar = document.getElementById('projects-sidebar');
            const mobileOverlay = document.getElementById('mobile-overlay');
            if (projectsSidebar) {
                projectsSidebar.classList.remove('mobile-sidebar-open');
                projectsSidebar.classList.add('mobile-sidebar-closed');
            }
            if (mobileOverlay) {
                mobileOverlay.classList.add('hidden');
            }
        }
        
        // Scroll to top of main content area
        const mainContent = document.querySelector('main .flex-1.overflow-y-auto');
        if (mainContent) {
            mainContent.scrollTop = 0;
        }
        
        // Load active jobs for this project
        loadProjectJobs(projectNumber);
        
        // Load audio files for alternate audio section
        if (typeof loadAudioFiles === 'function') {
            await loadAudioFiles();
        }
    } catch (error) {
        console.error('Failed to load project:', error);
        showToast('Failed to load project details', 'error');
    }
}

/**
 * Update project header with current project info
 */
function updateProjectHeader() {
    const detailsToggle = document.getElementById('project-details-toggle');
    const deleteBtn = document.getElementById('delete-project-btn');
    
    const subtitleEl = document.getElementById('current-project-subtitle');
    
    if (!currentProject) {
        document.getElementById('current-project-name').textContent = 'Select a project or upload new audio';
        subtitleEl.classList.add('hidden');
        detailsToggle.classList.add('hidden');
        return;
    }
    
    document.getElementById('current-project-name').textContent = `#${currentProject.number}: ${currentProject.name}`;
    
    const statuses = [];
    if (currentProject.files.stems.length > 0) statuses.push('stems separated');
    if (currentProject.files.cleaned.length > 0) statuses.push('stems cleaned');
    if (currentProject.files.midi.length > 0) statuses.push('MIDI generated');
    if (currentProject.files.video.length > 0) statuses.push('video rendered');
    
    subtitleEl.textContent = statuses.length > 0 ? statuses.join(' • ') : 'Ready to process';
    subtitleEl.classList.remove('hidden');
    
    // Show details toggle (delete button is now inside the details dropdown)
    detailsToggle.classList.remove('hidden');
    
    // Update project details
    const audioFile = currentProject.files.audio[0] || 'Unknown';
    document.getElementById('project-audio-file').textContent = audioFile;
    
    // Format created date if available
    if (currentProject.metadata && currentProject.metadata.created) {
        const date = new Date(currentProject.metadata.created);
        // Format with user's timezone
        const options = {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            timeZoneName: 'short'
        };
        document.getElementById('project-created').textContent = date.toLocaleString(undefined, options);
    } else {
        document.getElementById('project-created').textContent = 'Unknown';
    }
}

/**
 * Update operation button states based on project status
 */
function updateOperationButtons() {
    if (!currentProject) return;
    
    const files = currentProject.files;
    
    // Separate: Always enabled if we have audio
    const separateBtn = document.getElementById('btn-separate');
    separateBtn.disabled = files.audio.length === 0;
    
    // Cleanup: Requires stems
    const cleanupBtn = document.getElementById('btn-cleanup');
    cleanupBtn.disabled = files.stems.length === 0;
    
    // MIDI: Requires stems or cleaned stems
    const midiBtn = document.getElementById('btn-midi');
    midiBtn.disabled = files.stems.length === 0 && files.cleaned.length === 0;
    
    // Video: Requires MIDI
    const videoBtn = document.getElementById('btn-video');
    videoBtn.disabled = files.midi.length === 0;
}

/**
 * Update downloads section
 */
function updateDownloads() {
    if (!currentProject) return;
    
    const container = document.getElementById('downloads-list');
    const files = currentProject.files;
    const downloads = [];
    
    // Stems (with individual file downloads)
    if (files.stems.length > 0) {
        downloads.push({
            label: 'Stems',
            icon: 'fa-divide',
            color: 'blue',
            type: 'stems',
            count: files.stems.length,
            files: files.stems,
            hasIndividual: true
        });
    }
    
    // Cleaned (with individual file downloads)
    if (files.cleaned.length > 0) {
        downloads.push({
            label: 'Cleaned Stems',
            icon: 'fa-broom',
            color: 'purple',
            type: 'cleaned',
            count: files.cleaned.length,
            files: files.cleaned,
            hasIndividual: true
        });
    }
    
    // MIDI
    if (files.midi.length > 0) {
        downloads.push({
            label: 'MIDI',
            icon: 'fa-music',
            color: 'green',
            type: 'midi',
            count: files.midi.length
        });
    }
    
    // Video
    if (files.video.length > 0) {
        downloads.push({
            label: 'Video',
            icon: 'fa-video',
            color: 'orange',
            type: 'video',
            count: files.video.length
        });
    }
    
    if (downloads.length === 0) {
        container.innerHTML = '<p class="text-sm text-gray-500 col-span-full text-center py-4">No files available yet</p>';
        return;
    }
    
    container.innerHTML = downloads.map(dl => {
        if (dl.hasIndividual) {
            // Sort files in standard drum kit order and generate buttons
            const sortedFiles = sortDrumFiles(dl.files);
            const individualFiles = sortedFiles.map(file => {
                const drumName = file.split('-').pop().replace('.wav', '');
                const drumIcon = getDrumIcon(drumName);
                return `
                    <div class="flex items-center gap-1">
                        <button class="text-xs px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-1 transition-smooth flex-1"
                                onclick="event.stopPropagation(); playAudio('${dl.type}', '${file}')" 
                                title="Play ${drumName}">
                            <i class="fas ${drumIcon} text-${dl.color}-400"></i>
                            <span class="flex-1 text-left truncate">${drumName}</span>
                            <i class="fas fa-play text-xs opacity-60"></i>
                        </button>
                        <button class="text-xs px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-smooth"
                                onclick="event.stopPropagation(); downloadIndividualFile('${dl.type}', '${file}')" 
                                title="Download ${drumName}">
                            <i class="fas fa-download text-xs opacity-60"></i>
                        </button>
                    </div>
                `;
            }).join('');
            
            return `
                <div class="bg-gray-800 rounded-lg overflow-hidden border-2 border-${dl.color}-500">
                    <button class="glassy-btn w-full p-4 bg-${dl.color}-600 hover:bg-${dl.color}-700 text-white transition-smooth flex items-center justify-between font-semibold"
                            onclick="downloadFiles('${dl.type}')">
                        <div class="flex items-center">
                            <i class="fas fa-file-archive text-lg mr-3"></i>
                            <div class="text-left">
                                <div>${dl.label} (ZIP)</div>
                                <div class="text-xs font-normal opacity-75">Download all ${dl.count} files</div>
                            </div>
                        </div>
                        <i class="fas fa-download text-lg"></i>
                    </button>
                    <div class="bg-gray-800 p-3 border-t border-gray-700">
                        <div class="text-xs text-gray-400 mb-2 px-1 font-medium">Click to play or download individual files:</div>
                        <div class="grid grid-cols-2 gap-1.5">
                            ${individualFiles}
                        </div>
                    </div>
                </div>
            `;
        } else if (dl.type === 'midi') {
            // MIDI - just download
            return `
                <button class="glassy-btn bg-${dl.color}-600 hover:bg-${dl.color}-700 text-white p-3 rounded-lg border-2 border-${dl.color}-500 transition-smooth flex items-center justify-between"
                        onclick="downloadFiles('${dl.type}')">
                    <div>
                        <i class="fas ${dl.icon} mr-2"></i>
                        ${dl.label}
                    </div>
                    <i class="fas fa-download"></i>
                </button>
            `;
        } else if (dl.type === 'video') {
            // Video - play and download
            const videoFile = files.video[0];
            return `
                <div class="bg-gray-800 rounded-lg overflow-hidden border-2 border-${dl.color}-500">
                    <button class="glassy-btn w-full p-4 bg-${dl.color}-600 hover:bg-${dl.color}-700 text-white transition-smooth flex items-center justify-between font-semibold"
                            onclick="playVideo('${videoFile}')">
                        <div class="flex items-center">
                            <i class="fas fa-play-circle text-lg mr-3"></i>
                            <div class="text-left">
                                <div>${dl.label}</div>
                                <div class="text-xs font-normal opacity-75">Click to play</div>
                            </div>
                        </div>
                        <i class="fas fa-play text-lg"></i>
                    </button>
                    <div class="bg-gray-800 p-3 border-t border-gray-700">
                        <button class="w-full text-sm px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-smooth"
                                onclick="downloadFiles('${dl.type}')">
                            <i class="fas fa-download mr-2"></i>Download Video
                        </button>
                    </div>
                </div>
            `;
        }
    }).join('');
}

function getDrumIcon(drumName) {
    const icons = {
        'kick': 'fa-circle',
        'snare': 'fa-drumstick-bite',
        'hihat': 'fa-hat-wizard',
        'cymbals': 'fa-compact-disc',
        'toms': 'fa-drum'
    };
    return icons[drumName.toLowerCase()] || 'fa-music';
}

/**
 * Sort drum files in standard drum kit order
 */
function sortDrumFiles(files) {
    const drumOrder = ['kick', 'snare', 'toms', 'hihat', 'cymbals'];
    
    return files.slice().sort((a, b) => {
        const drumA = a.split('-').pop().replace('.wav', '').toLowerCase();
        const drumB = b.split('-').pop().replace('.wav', '').toLowerCase();
        
        const indexA = drumOrder.indexOf(drumA);
        const indexB = drumOrder.indexOf(drumB);
        
        // If both are in the order list, sort by position
        if (indexA !== -1 && indexB !== -1) {
            return indexA - indexB;
        }
        // If only A is in the list, it comes first
        if (indexA !== -1) return -1;
        // If only B is in the list, it comes first
        if (indexB !== -1) return 1;
        // Otherwise, sort alphabetically
        return drumA.localeCompare(drumB);
    });
}

function downloadIndividualFile(fileType, filename) {
    if (!currentProject) return;
    
    try {
        // Trigger download
        const url = `/api/projects/${currentProject.number}/download/${fileType}/${encodeURIComponent(filename)}`;
        window.location.href = url;
        
        showToast(`Downloading ${filename.split('-').pop().replace('.wav', '')}...`, 'success');
        
    } catch (error) {
        console.error('Download failed:', error);
        showToast('Download failed: ' + error.message, 'error');
    }
}

/**
 * Load active jobs for current project
 */
async function loadProjectJobs(projectNumber) {
    try {
        const data = await api.getProjectJobs(projectNumber);
        const activeJobs = data.jobs.filter(job => {
            const status = job.status.toLowerCase();
            return status === 'queued' || status === 'running';
        });
        
        if (activeJobs.length > 0) {
            document.getElementById('active-jobs-section').classList.remove('hidden');
            renderActiveJobs(activeJobs);
        } else {
            document.getElementById('active-jobs-section').classList.add('hidden');
        }
    } catch (error) {
        console.error('Failed to load project jobs:', error);
    }
}

/**
 * Render active jobs list
 */
function renderActiveJobs(jobs) {
    const container = document.getElementById('active-jobs-list');
    container.innerHTML = jobs.map(job => createJobCard(job)).join('');
}

/**
 * Download all files of a specific type
 */
async function downloadFiles(fileType) {
    if (!currentProject) return;
    
    try {
        // Show loading state
        showToast('Preparing download...', 'info');
        
        // Trigger download
        const url = `/api/projects/${currentProject.number}/download/${fileType}`;
        window.location.href = url;
        
        // Show success after a brief delay
        setTimeout(() => {
            showToast('Download started', 'success');
        }, 500);
        
    } catch (error) {
        console.error('Download failed:', error);
        showToast('Download failed: ' + error.message, 'error');
    }
}

/**
 * Toggle project details expanded/collapsed
 */
function toggleProjectDetails() {
    const container = document.getElementById('project-details-container');
    const icon = document.querySelector('#project-details-toggle i');
    const isCollapsed = container.style.maxHeight === '0px' || container.style.maxHeight === '';
    
    if (isCollapsed) {
        container.style.maxHeight = '100px';
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
    } else {
        container.style.maxHeight = '0';
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
    }
}

/**
 * Confirm and delete current project
 */
async function confirmDeleteProject() {
    if (!currentProject) return;
    
    const projectName = currentProject.name;
    const confirmed = confirm(
        `⚠️ WARNING: Delete Project?\n\n` +
        `This will permanently delete:\n` +
        `• Project: #${currentProject.number} - ${projectName}\n` +
        `• All audio files\n` +
        `• All stems\n` +
        `• All MIDI files\n` +
        `• All video files\n` +
        `• All configuration files\n\n` +
        `This action CANNOT be undone!\n\n` +
        `Are you absolutely sure?`
    );
    
    if (!confirmed) return;
    
    // Second confirmation
    const doubleConfirmed = confirm(
        `⚠️ FINAL WARNING\n\n` +
        `You are about to permanently delete project:\n` +
        `"${projectName}"\n\n` +
        `Click OK to DELETE FOREVER, or Cancel to keep the project.`
    );
    
    if (!doubleConfirmed) return;
    
    try {
        await api.deleteProject(currentProject.number);
        showToast(`Project "${projectName}" deleted`, 'success');
        
        // Clear current project
        currentProject = null;
        window.currentProject = null;
        
        // Hide project section and console, show landing
        document.getElementById('landing-section').classList.remove('hidden');
        document.getElementById('project-section').classList.add('hidden');
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('console-section').classList.add('hidden');
        
        // Reload projects list
        await loadProjects();
        
        // Update header
        updateProjectHeader();
    } catch (error) {
        console.error('Failed to delete project:', error);
        showToast(`Failed to delete project: ${error.message}`, 'error');
    }
}

/**
 * Play an audio file (WAV) in a modal
 */
function playAudio(fileType, filename) {
    if (!currentProject) return;
    
    const url = `/api/projects/${currentProject.number}/download/${fileType}/${encodeURIComponent(filename)}`;
    const drumName = filename.split('-').pop().replace('.wav', '');
    
    showMediaPlayer('audio', url, drumName);
}

/**
 * Play a video file in a modal
 */
function playVideo(filename) {
    if (!currentProject) return;
    
    const url = `/api/projects/${currentProject.number}/download/video/${encodeURIComponent(filename)}`;
    
    showMediaPlayer('video', url, filename);
}

/**
 * Show media player modal
 */
function showMediaPlayer(type, url, title) {
    // Create or get modal
    let modal = document.getElementById('media-player-modal');
    
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'media-player-modal';
        modal.className = 'fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4';
        modal.innerHTML = `
            <div class="bg-larsnet-dark rounded-lg border border-gray-700 max-w-4xl w-full">
                <div class="flex items-center justify-between p-4 border-b border-gray-700">
                    <h3 id="media-player-title" class="text-lg font-semibold"></h3>
                    <button onclick="closeMediaPlayer()" class="text-gray-400 hover:text-gray-200 transition-smooth">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                <div id="media-player-content" class="p-6">
                    <!-- Media element will be inserted here -->
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on click outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeMediaPlayer();
            }
        });
    }
    
    // Update title
    document.getElementById('media-player-title').textContent = title;
    
    // Create media element
    const content = document.getElementById('media-player-content');
    
    if (type === 'audio') {
        content.innerHTML = `
            <div class="flex flex-col items-center space-y-4">
                <div class="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                    <i class="fas fa-volume-up text-6xl text-white opacity-90"></i>
                </div>
                <audio id="media-player-element" controls preload="auto" class="w-full max-w-md">
                    <source src="${url}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
                <p id="media-player-error" class="text-sm text-red-400 hidden"></p>
            </div>
        `;
    } else if (type === 'video') {
        content.innerHTML = `
            <div class="relative flex items-center justify-center" style="max-height: 85vh;">
                <video id="media-player-element" controls preload="metadata" class="rounded bg-black" style="max-height: 85vh; max-width: 100%; object-fit: contain;">
                    <source src="${url}" type="video/mp4">
                    Your browser does not support video playback or the video format is incompatible.
                </video>
                <div id="media-loading" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded pointer-events-none">
                    <div class="text-center">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-larsnet-primary mx-auto mb-2"></div>
                        <p class="text-sm text-gray-300">Loading video...</p>
                    </div>
                </div>
                <p id="media-player-error" class="text-sm text-red-400 mt-2 hidden"></p>
            </div>
        `;
    }
    
    // Show modal
    modal.classList.remove('hidden');
    
    // Add event listeners for media element
    const mediaElement = document.getElementById('media-player-element');
    const loadingOverlay = document.getElementById('media-loading');
    const errorEl = document.getElementById('media-player-error');
    
    if (mediaElement) {
        // Debug logging
        console.log('Media player initialized:', type, url);
        console.log('Media element readyState:', mediaElement.readyState);
        console.log('Media element networkState:', mediaElement.networkState);
        
        // Handle various loading events
        mediaElement.addEventListener('loadstart', () => {
            console.log('Media load started');
        });
        
        mediaElement.addEventListener('loadedmetadata', () => {
            console.log('Media metadata loaded');
        });
        
        mediaElement.addEventListener('loadeddata', () => {
            console.log('Media data loaded');
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
            // Try to play (autoplay may be blocked by browser)
            mediaElement.play().catch(err => {
                console.log('Autoplay prevented:', err);
            });
        });
        
        mediaElement.addEventListener('canplay', () => {
            console.log('Media can play');
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
        });
        
        mediaElement.addEventListener('canplaythrough', () => {
            console.log('Media can play through');
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
        });
        
        // Handle errors
        mediaElement.addEventListener('error', (e) => {
            console.error('Media error event:', e);
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
            if (errorEl) {
                const error = e.target.error;
                let errorMsg = 'Unknown error';
                if (error) {
                    errorMsg = `Code ${error.code}: `;
                    switch(error.code) {
                        case 1: errorMsg += 'MEDIA_ERR_ABORTED - Fetching aborted'; break;
                        case 2: errorMsg += 'MEDIA_ERR_NETWORK - Network error'; break;
                        case 3: errorMsg += 'MEDIA_ERR_DECODE - Decoding error'; break;
                        case 4: errorMsg += 'MEDIA_ERR_SRC_NOT_SUPPORTED - Format not supported'; break;
                    }
                }
                errorEl.textContent = `Failed to load ${type}: ${errorMsg}`;
                errorEl.classList.remove('hidden');
            }
        });
        
        // Handle network issues
        mediaElement.addEventListener('stalled', () => {
            console.log('Media loading stalled');
        });
        
        mediaElement.addEventListener('waiting', () => {
            console.log('Media waiting for data');
        });
        
        mediaElement.addEventListener('progress', () => {
            console.log('Media loading progress');
        });
        
        // Only set source if it wasn't already set via <source> tag
        if (type === 'audio') {
            mediaElement.src = url;
        }
        mediaElement.load();
        
        // Fallback: hide loading after 10 seconds
        setTimeout(() => {
            if (loadingOverlay && !loadingOverlay.classList.contains('hidden')) {
                console.warn('Loading timeout - hiding overlay');
                loadingOverlay.classList.add('hidden');
            }
        }, 10000);
    }
}

/**
 * Close media player modal
 */
function closeMediaPlayer() {
    const modal = document.getElementById('media-player-modal');
    if (!modal) return;
    
    // Stop playback
    const mediaElement = document.getElementById('media-player-element');
    if (mediaElement) {
        mediaElement.pause();
        mediaElement.currentTime = 0;
    }
    
    // Hide modal
    modal.classList.add('hidden');
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
