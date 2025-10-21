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
        showLoading('Loading project...');
        
        const data = await api.getProject(projectNumber);
        currentProject = data.project;
        
        renderProjectsList(); // Re-render to update active state
        updateProjectHeader();
        updateOperationButtons();
        updateDownloads();
        
        // Show project section and sidebar upload, hide main upload
        document.getElementById('project-section').classList.remove('hidden');
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('sidebar-upload-section').classList.remove('hidden');
        
        // Load active jobs for this project
        loadProjectJobs(projectNumber);
        
        hideLoading();
    } catch (error) {
        hideLoading();
        console.error('Failed to load project:', error);
        showToast('Failed to load project details', 'error');
    }
}

/**
 * Update project header with current project info
 */
function updateProjectHeader() {
    if (!currentProject) {
        document.getElementById('current-project-name').textContent = 'Select a project or upload new audio';
        document.getElementById('current-project-subtitle').textContent = 'Drag and drop an audio file to get started';
        return;
    }
    
    document.getElementById('current-project-name').textContent = `#${currentProject.number}: ${currentProject.name}`;
    
    const statuses = [];
    if (currentProject.files.stems.length > 0) statuses.push('stems separated');
    if (currentProject.files.cleaned.length > 0) statuses.push('stems cleaned');
    if (currentProject.files.midi.length > 0) statuses.push('MIDI generated');
    if (currentProject.files.video.length > 0) statuses.push('video rendered');
    
    document.getElementById('current-project-subtitle').textContent = 
        statuses.length > 0 ? statuses.join(' • ') : 'Ready to process';
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
    
    // Stems
    if (files.stems.length > 0) {
        downloads.push({
            label: 'Stems',
            icon: 'fa-divide',
            color: 'blue',
            type: 'stems',
            count: files.stems.length
        });
    }
    
    // Cleaned
    if (files.cleaned.length > 0) {
        downloads.push({
            label: 'Cleaned Stems',
            icon: 'fa-broom',
            color: 'purple',
            type: 'cleaned',
            count: files.cleaned.length
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
    
    container.innerHTML = downloads.map(dl => `
        <button class="bg-${dl.color}-600 hover:bg-${dl.color}-700 text-white p-3 rounded-lg transition-smooth flex items-center justify-between"
                onclick="downloadFiles('${dl.type}')">
            <div>
                <i class="fas ${dl.icon} mr-2"></i>
                ${dl.label}
            </div>
            <span class="text-xs opacity-75">${dl.count}</span>
        </button>
    `).join('');
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
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
