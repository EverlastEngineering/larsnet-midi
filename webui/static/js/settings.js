/**
 * Settings Panel Manager
 * 
 * Manages collapsible settings panels for each operation.
 * Handles toggle animations, localStorage persistence, and value retrieval.
 */

class SettingsManager {
    constructor() {
        this.currentlyOpen = null;
        this.settings = this.loadSettings();
        this.init();
    }
    
    /**
     * Initialize settings manager
     */
    init() {
        // Bind toggle buttons
        const toggleButtons = document.querySelectorAll('.settings-toggle');
        toggleButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const target = btn.dataset.target;
                this.togglePanel(target);
            });
        });
        
        // Bind input changes to localStorage
        this.bindSettingsInputs();
        
        // Restore saved settings
        this.restoreSettings();
    }
    
    /**
     * Toggle a settings panel open/closed
     */
    togglePanel(panelId) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        
        const isCurrentlyOpen = !panel.classList.contains('hidden');
        
        // Close all panels first
        document.querySelectorAll('.settings-panel').forEach(p => {
            p.classList.add('hidden');
        });
        
        // Reset all toggle button icons
        document.querySelectorAll('.settings-toggle i').forEach(icon => {
            icon.className = 'fas fa-chevron-down mr-1';
        });
        
        // If panel was closed, open it
        if (!isCurrentlyOpen) {
            panel.classList.remove('hidden');
            this.currentlyOpen = panelId;
            
            // Update button icon to chevron-up
            const button = document.querySelector(`[data-target="${panelId}"] i`);
            if (button) {
                button.className = 'fas fa-chevron-up mr-1';
            }
            
            // Smooth scroll into view
            setTimeout(() => {
                panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        } else {
            this.currentlyOpen = null;
        }
    }
    
    /**
     * Bind all settings inputs to save to localStorage on change
     */
    bindSettingsInputs() {
        // All inputs with IDs starting with "setting-"
        const inputs = document.querySelectorAll('[id^="setting-"]');
        
        inputs.forEach(input => {
            const settingKey = input.id.replace('setting-', '');
            
            input.addEventListener('change', () => {
                const value = this.getInputValue(input);
                this.settings[settingKey] = value;
                this.saveSettings();
            });
        });
    }
    
    /**
     * Get value from input element (handles different input types)
     */
    getInputValue(input) {
        if (input.type === 'checkbox') {
            return input.checked;
        } else if (input.type === 'number') {
            return input.value === '' ? null : parseFloat(input.value);
        } else {
            return input.value;
        }
    }
    
    /**
     * Set value to input element (handles different input types)
     */
    setInputValue(input, value) {
        if (input.type === 'checkbox') {
            input.checked = value === true;
        } else if (value !== null && value !== undefined) {
            input.value = value;
        }
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        try {
            const saved = localStorage.getItem('larsnet_settings');
            return saved ? JSON.parse(saved) : {};
        } catch (e) {
            console.error('Failed to load settings:', e);
            return {};
        }
    }
    
    /**
     * Save settings to localStorage
     */
    saveSettings() {
        try {
            localStorage.setItem('larsnet_settings', JSON.stringify(this.settings));
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
    }
    
    /**
     * Restore settings from localStorage to UI inputs
     */
    restoreSettings() {
        Object.keys(this.settings).forEach(key => {
            const input = document.getElementById(`setting-${key}`);
            if (input) {
                this.setInputValue(input, this.settings[key]);
            }
        });
    }
    
    /**
     * Get settings for a specific operation
     */
    getSettingsForOperation(operation) {
        const settings = {};
        
        switch (operation) {
            case 'separate':
                settings.device = this.settings['device'] || 'cpu';
                settings.apply_eq = this.settings['apply-eq'] || false;
                settings.wiener_exponent = this.settings['wiener'] || null;
                if (settings.wiener_exponent === 0) {
                    settings.wiener_exponent = null; // 0 means disabled
                }
                break;
                
            case 'cleanup':
                settings.threshold = this.settings['threshold'] || -30.0;
                settings.ratio = this.settings['ratio'] || 10.0;
                settings.attack = this.settings['attack'] || 1.0;
                settings.release = this.settings['release'] || 100.0;
                break;
                
            case 'midi':
                settings.onset_threshold = this.settings['onset-threshold'] || null;
                settings.onset_delta = this.settings['onset-delta'] || null;
                settings.min_velocity = this.settings['min-velocity'] || 40;
                settings.max_velocity = this.settings['max-velocity'] || 127;
                settings.tempo = this.settings['tempo'] || null;
                settings.detect_hihat_open = this.settings['detect-hihat-open'] || false;
                break;
                
            case 'video':
                settings.fps = this.settings['fps'] || 60;
                settings.resolution = this.settings['resolution'] || '1080p';
                settings.audioSource = this.settings['audio-source'] || '';
                settings.fallSpeed = this.settings['fall-speed'] || 1.0;
                break;
        }
        
        return settings;
    }
    
    /**
     * Reset settings to defaults for a specific operation
     */
    resetOperation(operation) {
        const defaults = {
            'separate': {
                'device': 'cpu',
                'apply-eq': false,
                'wiener': 0
            },
            'cleanup': {
                'threshold': -30.0,
                'ratio': 10.0,
                'attack': 1.0,
                'release': 100.0
            },
            'midi': {
                'onset-threshold': 0.3,
                'onset-delta': 0.01,
                'min-velocity': 40,
                'max-velocity': 127,
                'tempo': null,
                'detect-hihat-open': false
            },
            'video': {
                'fps': 60,
                'resolution': '1080p',
                'audio-source': '',
                'fall-speed': 1.0
            }
        };
        
        const operationDefaults = defaults[operation] || {};
        Object.keys(operationDefaults).forEach(key => {
            this.settings[key] = operationDefaults[key];
            const input = document.getElementById(`setting-${key}`);
            if (input) {
                this.setInputValue(input, operationDefaults[key]);
            }
        });
        
        this.saveSettings();
    }
    
    /**
     * Get all current settings as a flat object
     */
    getAllSettings() {
        // Refresh from current input values
        const inputs = document.querySelectorAll('[id^="setting-"]');
        inputs.forEach(input => {
            const settingKey = input.id.replace('setting-', '');
            this.settings[settingKey] = this.getInputValue(input);
        });
        
        return { ...this.settings };
    }
}

// Export for use in operations.js
window.SettingsManager = SettingsManager;
