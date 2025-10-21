/**
 * Advanced MIDI Settings UI
 * 
 * Dynamically renders configuration UI from YAML structure using the config engine.
 * Provides per-stem control over onset detection, spectral filtering, and timing.
 */

class AdvancedMIDISettings {
    constructor() {
        this.modal = document.getElementById('advanced-midi-modal');
        this.content = document.getElementById('advanced-midi-content');
        this.currentProject = null;
        this.configData = null;
        this.changes = new Map(); // Track changes: path -> value
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        // Open modal button
        document.getElementById('btn-advanced-midi')?.addEventListener('click', () => {
            this.open();
        });
        
        // Close buttons
        document.getElementById('close-advanced-midi')?.addEventListener('click', () => {
            this.close();
        });
        
        document.getElementById('cancel-advanced-midi')?.addEventListener('click', () => {
            this.close();
        });
        
        // Click outside to close
        this.modal?.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });
        
        // Save button
        document.getElementById('save-advanced-midi')?.addEventListener('click', () => {
            this.save();
        });
        
        // Reset button
        document.getElementById('reset-advanced-midi')?.addEventListener('click', () => {
            this.reset();
        });
    }
    
    async open() {
        // Get current project
        this.currentProject = window.currentProject;
        
        if (!this.currentProject) {
            showToast('Please select a project first', 'warning');
            return;
        }
        
        // Show modal
        this.modal.classList.remove('hidden');
        
        // Load configuration
        await this.loadConfig();
    }
    
    close() {
        this.modal.classList.add('hidden');
        this.changes.clear();
    }
    
    async loadConfig() {
        try {
            this.showLoading();
            
            const response = await fetch(`/api/config/${this.currentProject.number}/midiconfig`);
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to load configuration');
            }
            
            this.configData = data;
            this.renderConfig(data.sections);
            
        } catch (error) {
            console.error('Failed to load MIDI config:', error);
            this.showError(error.message);
        }
    }
    
    showLoading() {
        this.content.innerHTML = `
            <div class="flex items-center justify-center py-12">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-larsnet-primary"></div>
                <span class="ml-4 text-gray-400">Loading configuration...</span>
            </div>
        `;
    }
    
    showError(message) {
        this.content.innerHTML = `
            <div class="flex flex-col items-center justify-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-larsnet-error mb-4"></i>
                <p class="text-gray-400">${message}</p>
                <button onclick="location.reload()" class="mt-4 px-4 py-2 bg-larsnet-primary hover:bg-blue-600 rounded transition-smooth">
                    Reload Page
                </button>
            </div>
        `;
    }
    
    renderConfig(sections) {
        if (!sections || sections.length === 0) {
            this.showError('No configuration sections found');
            return;
        }
        
        // Group sections by type
        const audioSection = sections.find(s => s.name === 'audio');
        const onsetSection = sections.find(s => s.name === 'onset_detection');
        const stemSections = sections.filter(s => 
            ['kick', 'snare', 'toms', 'hihat', 'cymbals'].includes(s.name)
        );
        const midiSection = sections.find(s => s.name === 'midi');
        const debugSection = sections.find(s => s.name === 'debug');
        const learningSection = sections.find(s => s.name === 'learning_mode');
        
        let html = '<div class="space-y-8">';
        
        // Global Settings
        if (audioSection || onsetSection || midiSection) {
            html += '<div class="space-y-6">';
            html += '<h3 class="text-xl font-bold text-larsnet-primary border-b border-gray-700 pb-2">Global Settings</h3>';
            
            if (audioSection) {
                html += this.renderSection(audioSection, false);
            }
            
            if (onsetSection) {
                html += this.renderSection(onsetSection, false);
            }
            
            if (midiSection) {
                html += this.renderSection(midiSection, false);
            }
            
            html += '</div>';
        }
        
        // Per-Stem Settings
        if (stemSections.length > 0) {
            html += '<div class="space-y-6">';
            html += '<h3 class="text-xl font-bold text-larsnet-primary border-b border-gray-700 pb-2">Per-Stem Settings</h3>';
            
            for (const section of stemSections) {
                html += this.renderSection(section, true);
            }
            
            html += '</div>';
        }
        
        // Advanced Settings
        if (debugSection || learningSection) {
            html += '<div class="space-y-6">';
            html += '<h3 class="text-xl font-bold text-larsnet-primary border-b border-gray-700 pb-2">Advanced / Debug</h3>';
            
            if (debugSection) {
                html += this.renderSection(debugSection, false);
            }
            
            if (learningSection) {
                html += this.renderSection(learningSection, false);
            }
            
            html += '</div>';
        }
        
        html += '</div>';
        
        this.content.innerHTML = html;
        this.attachFieldListeners();
    }
    
    renderSection(section, collapsible = false) {
        const sectionId = `section-${section.name}`;
        
        let html = '<div class="bg-gray-800 rounded-lg p-6 border border-gray-700">';
        
        // Section header
        if (collapsible) {
            html += `
                <button class="w-full flex items-center justify-between text-left section-toggle" data-section="${sectionId}">
                    <div>
                        <h4 class="text-lg font-semibold capitalize">${section.label}</h4>
                        ${section.description ? `<p class="text-sm text-gray-400 mt-1">${section.description}</p>` : ''}
                    </div>
                    <i class="fas fa-chevron-down text-gray-400 transition-transform"></i>
                </button>
                <div id="${sectionId}" class="section-content mt-4 space-y-4 hidden">
            `;
        } else {
            html += `
                <h4 class="text-lg font-semibold capitalize mb-4">${section.label}</h4>
                ${section.description ? `<p class="text-sm text-gray-400 mb-4">${section.description}</p>` : ''}
                <div class="space-y-4">
            `;
        }
        
        // Render fields
        for (const field of section.fields) {
            html += this.renderField(field);
        }
        
        html += '</div></div>';
        
        return html;
    }
    
    renderField(field) {
        const fieldId = `field-${field.path.replace(/\./g, '-')}`;
        const isNull = field.value === null;
        
        let html = '<div class="field-container">';
        
        // Label
        html += `
            <label for="${fieldId}" class="text-sm font-medium text-gray-300 block mb-2">
                ${field.label}
            </label>
        `;
        
        // Description
        if (field.description) {
            html += `<p class="text-xs text-gray-500 mb-2">${field.description}</p>`;
        }
        
        // Input based on type
        switch (field.type) {
            case 'bool':
                html += this.renderBoolField(field, fieldId);
                break;
            case 'int':
            case 'float':
                html += this.renderNumberField(field, fieldId, isNull);
                break;
            case 'string':
            case 'path':
                html += this.renderTextFields(field, fieldId, isNull);
                break;
            default:
                html += this.renderTextFields(field, fieldId, isNull);
        }
        
        html += '</div>';
        
        return html;
    }
    
    renderBoolField(field, fieldId) {
        const checked = field.value ? 'checked' : '';
        return `
            <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" 
                       id="${fieldId}" 
                       data-path="${field.path}"
                       data-type="${field.type}"
                       ${checked}
                       class="sr-only peer config-field">
                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-larsnet-primary"></div>
            </label>
        `;
    }
    
    renderNumberField(field, fieldId, isNull) {
        const min = field.validation?.min !== null ? field.validation.min : '';
        const max = field.validation?.max !== null ? field.validation.max : '';
        const step = field.type === 'float' ? '0.001' : '1';
        const value = isNull ? '' : field.value;
        const placeholder = isNull ? 'Use global default' : '';
        
        return `
            <div class="flex items-center gap-2">
                <input type="number" 
                       id="${fieldId}" 
                       data-path="${field.path}"
                       data-type="${field.type}"
                       data-nullable="${isNull}"
                       value="${value}"
                       placeholder="${placeholder}"
                       ${min !== '' ? `min="${min}"` : ''}
                       ${max !== '' ? `max="${max}"` : ''}
                       step="${step}"
                       class="bg-gray-700 text-gray-200 rounded px-3 py-2 text-sm w-48 config-field">
                ${isNull ? '<span class="text-xs text-gray-500">(null = use global setting)</span>' : ''}
            </div>
        `;
    }
    
    renderTextFields(field, fieldId, isNull) {
        const value = isNull ? '' : field.value;
        const placeholder = isNull ? 'Use global default' : '';
        
        return `
            <input type="text" 
                   id="${fieldId}" 
                   data-path="${field.path}"
                   data-type="${field.type}"
                   data-nullable="${isNull}"
                   value="${value}"
                   placeholder="${placeholder}"
                   class="bg-gray-700 text-gray-200 rounded px-3 py-2 text-sm w-full max-w-md config-field">
        `;
    }
    
    attachFieldListeners() {
        // Section toggle
        document.querySelectorAll('.section-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const sectionId = btn.getAttribute('data-section');
                const section = document.getElementById(sectionId);
                const icon = btn.querySelector('i');
                
                section.classList.toggle('hidden');
                icon.classList.toggle('rotate-180');
            });
        });
        
        // Field changes
        document.querySelectorAll('.config-field').forEach(field => {
            field.addEventListener('change', (e) => {
                this.handleFieldChange(e.target);
            });
        });
    }
    
    handleFieldChange(field) {
        const path = field.getAttribute('data-path').split('.');
        const type = field.getAttribute('data-type');
        let value;
        
        // Parse value based on type
        switch (type) {
            case 'bool':
                value = field.checked;
                break;
            case 'int':
                value = field.value === '' ? null : parseInt(field.value, 10);
                break;
            case 'float':
                value = field.value === '' ? null : parseFloat(field.value);
                break;
            default:
                value = field.value === '' ? null : field.value;
        }
        
        // Track change
        this.changes.set(path.join('.'), { path, value });
        
        // Visual feedback
        field.classList.add('border-larsnet-warning', 'border-2');
    }
    
    async save() {
        if (this.changes.size === 0) {
            showToast('No changes to save', 'info');
            return;
        }
        
        const updates = Array.from(this.changes.values());
        
        try {
            // Show saving state
            const saveBtn = document.getElementById('save-advanced-midi');
            const originalText = saveBtn.innerHTML;
            saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Saving...';
            saveBtn.disabled = true;
            
            // Save to server
            const response = await fetch(`/api/config/${this.currentProject.number}/midiconfig`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ updates })
            });
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to save configuration');
            }
            
            showToast('MIDI configuration saved successfully', 'success');
            this.changes.clear();
            
            // Remove visual feedback from fields
            document.querySelectorAll('.config-field').forEach(field => {
                field.classList.remove('border-larsnet-warning', 'border-2');
            });
            
            // Restore button
            saveBtn.innerHTML = originalText;
            saveBtn.disabled = false;
            
        } catch (error) {
            console.error('Failed to save configuration:', error);
            showToast(`Failed to save: ${error.message}`, 'error');
            
            // Restore button
            const saveBtn = document.getElementById('save-advanced-midi');
            saveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>Save Changes';
            saveBtn.disabled = false;
        }
    }
    
    async reset() {
        if (!confirm('Reset all settings to defaults? This cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/config/${this.currentProject.number}/midiconfig/reset`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Failed to reset configuration');
            }
            
            showToast('MIDI configuration reset to defaults', 'success');
            
            // Reload config
            await this.loadConfig();
            this.changes.clear();
            
        } catch (error) {
            console.error('Failed to reset configuration:', error);
            showToast(`Failed to reset: ${error.message}`, 'error');
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.advancedMIDISettings = new AdvancedMIDISettings();
});
