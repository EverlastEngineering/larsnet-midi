/**
 * API Client for LarsNet Web UI
 * 
 * Provides a clean interface to all backend API endpoints with error handling.
 */

const API_BASE = window.location.origin + '/api';

/**
 * API client class
 */
class LarsNetAPI {
    
    /**
     * Make a GET request
     */
    async get(endpoint) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || data.error || 'Request failed');
            }
            
            return data;
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    }
    
    /**
     * Make a POST request
     */
    async post(endpoint, body = null) {
        try {
            const options = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (body) {
                options.body = JSON.stringify(body);
            }
            
            const response = await fetch(`${API_BASE}${endpoint}`, options);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || data.error || 'Request failed');
            }
            
            return data;
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    }
    
    /**
     * Upload a file
     */
    async upload(file, onProgress = null) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);
            
            const xhr = new XMLHttpRequest();
            
            // Progress tracking
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent);
                    }
                });
            }
            
            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        resolve(data);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.message || error.error || 'Upload failed'));
                    } catch {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                }
            });
            
            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });
            
            xhr.addEventListener('abort', () => {
                reject(new Error('Upload cancelled'));
            });
            
            xhr.open('POST', `${API_BASE}/upload`);
            xhr.send(formData);
        });
    }
    
    /**
     * Connect to Server-Sent Events for job updates
     */
    streamJobStatus(jobId, callbacks = {}) {
        const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/stream`);
        
        eventSource.addEventListener('job_update', (e) => {
            const job = JSON.parse(e.data);
            if (callbacks.onUpdate) callbacks.onUpdate(job);
        });
        
        eventSource.addEventListener('job_complete', (e) => {
            const job = JSON.parse(e.data);
            if (callbacks.onComplete) callbacks.onComplete(job);
            eventSource.close();
        });
        
        eventSource.addEventListener('job_error', (e) => {
            const job = JSON.parse(e.data);
            if (callbacks.onError) callbacks.onError(job);
            eventSource.close();
        });
        
        eventSource.addEventListener('error', (e) => {
            console.error('SSE connection error:', e);
            if (callbacks.onConnectionError) callbacks.onConnectionError(e);
            eventSource.close();
        });
        
        return eventSource;
    }
    
    // ========== Projects API ==========
    
    async getProjects() {
        return await this.get('/projects');
    }
    
    async getProject(projectNumber) {
        return await this.get(`/projects/${projectNumber}`);
    }
    
    async getProjectConfig(projectNumber, configName) {
        return await this.get(`/projects/${projectNumber}/config/${configName}`);
    }
    
    async getProjectJobs(projectNumber) {
        return await this.get(`/projects/${projectNumber}/jobs`);
    }
    
    async deleteProject(projectNumber) {
        try {
            const response = await fetch(`${API_BASE}/projects/${projectNumber}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || data.error || 'Delete failed');
            }
            
            return data;
        } catch (error) {
            console.error(`DELETE /projects/${projectNumber} failed:`, error);
            throw error;
        }
    }
    
    async getAudioFiles(projectNumber) {
        return await this.get(`/projects/${projectNumber}/audio-files`);
    }
    
    async uploadAlternateAudio(projectNumber, file, onProgress = null) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);
            
            const xhr = new XMLHttpRequest();
            
            // Progress tracking
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent);
                    }
                });
            }
            
            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        resolve(data);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.message || error.error || 'Upload failed'));
                    } catch {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                }
            });
            
            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });
            
            // Send request
            xhr.open('POST', `${API_BASE}/projects/${projectNumber}/upload-alternate-audio`);
            xhr.send(formData);
        });
    }
    
    async deleteAudioFile(projectNumber, filename) {
        try {
            const response = await fetch(`${API_BASE}/projects/${projectNumber}/audio-files/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || data.error || 'Delete failed');
            }
            
            return data;
        } catch (error) {
            console.error(`DELETE /projects/${projectNumber}/audio-files/${filename} failed:`, error);
            throw error;
        }
    }
    
    // ========== Operations API ==========
    
    async separate(projectNumber, options = {}) {
        return await this.post('/separate', {
            project_number: projectNumber,
            ...options
        });
    }
    
    async cleanup(projectNumber, options = {}) {
        return await this.post('/cleanup', {
            project_number: projectNumber,
            ...options
        });
    }
    
    async stemsToMidi(projectNumber, options = {}) {
        return await this.post('/stems-to-midi', {
            project_number: projectNumber,
            ...options
        });
    }
    
    async renderVideo(projectNumber, options = {}) {
        return await this.post('/render-video', {
            project_number: projectNumber,
            ...options
        });
    }
    
    // ========== Jobs API ==========
    
    async getJobs() {
        return await this.get('/jobs');
    }
    
    async getJob(jobId) {
        return await this.get(`/jobs/${jobId}`);
    }
    
    async cancelJob(jobId) {
        return await this.post(`/jobs/${jobId}/cancel`);
    }
    
    // ========== Health Check ==========
    
    async checkHealth() {
        try {
            const response = await fetch('/health');
            return response.ok;
        } catch {
            return false;
        }
    }
}

// Create global API instance
const api = new LarsNetAPI();
