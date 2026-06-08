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
        
        // Resolve coupled dependencies BEFORE bundling updates.
        // Example: setting snare.cluster_feature='pitch_hz' requires
        // snare.enable_pitch_detection=true (otherwise the pipeline
        // silently falls back — see stems_to_midi/note_classification_core.py).
        // We add the coupled toggle to the same payload so the user
        // makes one save instead of two.
        let updates = Array.from(this.changes.values());
        const dependencyNote = this._applyClusterFeatureDependencies(updates);
        
        // Detection-time keys (enable_pitch_detection, pitch_method,
        // pitch Hz ranges, etc.) only take effect on a full Convert —
        // rebuild alone re-uses the stored analysis.json. Surface
        // this so the user knows why their Save & Reconvert might
        // not visibly change anything.
        const needsFullConvert = this._requiresFullConvert(updates);
        
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
            
            // Reload configData so subsequent saves in this session
            // see the just-saved values (not the stale snapshot from
            // when the modal opened). Without this, the dependency
            // handler's "is enable_pitch_detection already true?"
            // check uses stale data and the auto-toggle fires
            // spuriously on every cluster_feature change.
            await this.loadConfig();
            
            // Remove visual feedback from fields
            document.querySelectorAll('.config-field').forEach(field => {
                field.classList.remove('border-larsnet-warning', 'border-2');
            });
            
            // Restore button
            saveBtn.innerHTML = originalText;
            saveBtn.disabled = false;
            
            // Inform the user about the coupled dependency we just
            // applied, and the fact that detection-time changes need
            // a full Convert. Both toasts are info-level — they
            // supplement the success toast above. The showToast
            // helper currently uses a hard-coded 5s display; if
            // longer-lived toasts become useful, add a duration
            // parameter there.
            if (dependencyNote) {
                showToast(dependencyNote, 'info');
            }
            if (needsFullConvert) {
                showToast(
                    'These changes require a full Convert (not just Save & Reconvert) to take effect. ' +
                    'Save & Reconvert reuses the stored analysis.json; Convert re-runs detection.',
                    'info'
                );
            }
            
        } catch (error) {
            console.error('Failed to save configuration:', error);
            showToast(`Failed to save: ${error.message}`, 'error');
            
            // Restore button
            const saveBtn = document.getElementById('save-advanced-midi');
            saveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>Save Changes';
            saveBtn.disabled = false;
        }
    }
    
    /**
     * Apply coupled cluster-feature dependencies to the updates list.
     *
     * The classic case (user report 2026-06-08): the user picks
     * "Pitch" in the snare Cluster By dropdown, but
     * ``enable_pitch_detection`` is false. The pipeline never
     * computes ``pitch_hz`` for snare onsets, and
     * ``_resolve_cluster_feature`` silently falls back to
     * ``stereo_width``. The user sees no change.
     *
     * Fix: when the user saves ``cluster_feature: pitch_hz`` on a
     * stem whose ``enable_pitch_detection`` is false, auto-add
     * ``enable_pitch_detection: true`` to the same payload. The
     * server applies both updates atomically.
     *
     * Returns a string to show in a toast (or null if no dependency
     * was applied). The caller surfaces the toast after the save
     * succeeds.
     */
    _applyClusterFeatureDependencies(updates) {
        // Read the parsed config to know the current value of
        // enable_pitch_detection. configData is set in loadConfig().
        // Also check this.changes — if the user has already toggled
        // enable_pitch_detection to true in this session, we must
        // not re-add it (avoids a redundant update and a misleading
        // dependency toast).
        //
        // The config payload uses DOTTED-STRING paths (e.g.
        // "snare.enable_pitch_detection") for the per-field metadata,
        // but the updates list uses LIST paths (e.g. ['snare',
        // 'enable_pitch_detection']). We have to handle both shapes.
        const configSections = (this.configData && this.configData.sections) || [];
        const stemEnablePitch = {};
        for (const section of configSections) {
            if (!['snare', 'toms', 'cymbals'].includes(section.name)) continue;
            for (const field of section.fields) {
                // field.path may be a string ('snare.enable_pitch_detection')
                // or a list (['snare', 'enable_pitch_detection']) depending
                // on the engine. Accept both.
                const parts = typeof field.path === 'string'
                    ? field.path.split('.')
                    : field.path;
                if (!parts || parts.length !== 2) continue;
                if (parts[0] === section.name && parts[1] === 'enable_pitch_detection') {
                    stemEnablePitch[section.name] = field.value === true;
                }
            }
        }
        // Overlay the user's pending changes on top of the loaded
        // config so we see the effective "current" value.
        for (const [, change] of this.changes) {
            const path = change.path;
            if (Array.isArray(path) && path.length === 2 && path[1] === 'enable_pitch_detection') {
                if (['snare', 'toms', 'cymbals'].includes(path[0])) {
                    stemEnablePitch[path[0]] = change.value === true;
                }
            }
        }
        
        const added = [];
        for (const update of updates) {
            if (!Array.isArray(update.path) || update.path.length !== 2) continue;
            const [stem, key] = update.path;
            if (key !== 'cluster_feature') continue;
            if (!['snare', 'toms', 'cymbals'].includes(stem)) continue;
            if (update.value !== 'pitch_hz') continue;
            
            // User picked pitch_hz. If the stem's enable_pitch_detection
            // is currently false (or missing), add a true update.
            if (stemEnablePitch[stem] === true) continue;
            
            updates.push({
                path: [stem, 'enable_pitch_detection'],
                value: true,
            });
            added.push(stem);
        }
        
        if (added.length === 0) return null;
        const stemList = added.join(', ');
        return `Pitch selected for ${stemList} — pitch detection enabled automatically. ` +
               `A full Convert is required to compute pitch data; Save & Reconvert alone won't update the analysis.`;
    }
    
    /**
     * Returns true if any update in the list touches a detection-time
     * key. Detection-time keys are computed during the full Convert
     * pipeline (audio re-processing) and are NOT re-computed during a
     * rebuild — the rebuild path only re-classifies from stored
     * features. If the user changed a detection-time key, they need
     * to run a full Convert for the change to take effect on the
     * stored analysis.json.
     */
    _requiresFullConvert(updates) {
        // Detection-time keys live at <stem>.* where * is one of:
        //   enable_pitch_detection
        //   pitch_method
        //   min_pitch_hz
        //   max_pitch_hz
        //   fundamental_freq_min
        //   fundamental_freq_max
        //   body_freq_min / body_freq_max
        //   low_freq_min / low_freq_max
        //   wire_freq_min / wire_freq_max
        //   enable_statistical_filter  (computed per-event)
        // We keep this conservative — false positives (warning when
        // not needed) are better than false negatives (silently
        // failing to apply a change).
        const detectionTimeLeaf = (key) => (
            key === 'enable_pitch_detection' ||
            key === 'pitch_method' ||
            key.endsWith('_pitch_hz') ||
            key.endsWith('_freq_min') ||
            key.endsWith('_freq_max')
        );
        
        for (const update of updates) {
            if (!Array.isArray(update.path) || update.path.length !== 2) continue;
            const [, key] = update.path;
            // cluster_feature changes are also "detection-time" in
            // the sense that they don't take effect on stored data
            // without re-detection. Bundle them into the same hint
            // so the user gets one consistent message.
            if (key === 'cluster_feature' || detectionTimeLeaf(key)) {
                return true;
            }
        }
        return false;
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
