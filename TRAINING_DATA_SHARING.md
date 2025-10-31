# Training Data Sharing & Universal Model Building

## Vision

Enable collaborative improvement of drum detection by allowing users to contribute anonymized training data to build a universal, high-accuracy model that benefits everyone.

## Why Share Training Data?

### Current Limitation
- Each user optimizes detection for their own songs
- Starting from scratch every time
- No cross-learning between users
- Small training sets (6-20 examples per song)

### With Data Sharing
- Aggregate thousands of labeled examples
- Train robust universal model
- New users get great results immediately
- Continuous improvement as more data arrives
- Rare patterns (ghost notes, flams) get learned

### Example Impact
```
Single User:
  6 songs × 6 examples = 36 training examples
  Accuracy: ~85-90% (limited generalization)

Community (1000 users):
  1000 songs × 6 examples = 6,000 training examples  
  Accuracy: ~97-99% (robust across styles)
  
  Bonus: Rare techniques represented
  - Soft open hi-hats: 50 examples (vs 0-1 for single user)
  - Half-open: 200 examples (vs 0 for single user)
  - Foot splash: 30 examples (vs 0 for single user)
```

## How It Works

### 1. Local Optimization (Always First)

User runs optimizer on their song:
```bash
python optimize_config.py --project 4 --instrument hihat
```

Gets custom thresholds that work perfectly for their project.

### 2. Export for Sharing (Optional)

After validation, user can export anonymized data:
```bash
python export_training_data.py --project 4 --instrument hihat
```

Creates a JSON file:
```json
{
  "version": "1.0",
  "instrument": "hihat",
  "features": [
    {"BodyE": 251.6, "SizzleE": 3264.1, "GeoMean": 906.2, "SustainMs": 181.6, "OpenHH": 1},
    {"BodyE": 37.4, "SizzleE": 359.4, "GeoMean": 116.0, "SustainMs": 26.6, "OpenHH": 0},
    ...
  ],
  "metadata": {
    "tempo": 124,
    "genre": "pop",
    "sample_count": 160,
    "open_count": 6,
    "timestamp": "2025-10-31T14:30:00Z"
  }
}
```

### 3. Review Before Sharing

User reviews the export:
- **No audio** - just numerical features
- **No filenames** - completely anonymous
- **No timestamps** - can't reconstruct timing
- **No user info** - privacy preserved

### 4. Submit to Community

```bash
python submit_training_data.py training_exports/hihat_20251031_143000.json
```

Or via web interface:
```
https://larsnet-midi.com/contribute
[Drag & Drop JSON file]
[Review data preview]
[Submit] [Cancel]
```

### 5. Aggregation & Training

Backend service:
1. Validates submission (format, ranges)
2. Adds to training pool
3. Periodically retrains universal model (weekly)
4. Publishes updated model for download

### 6. Download Updated Model

Users get automatic improvements:
```bash
python update_models.py --check
# New model available: hihat-v2.1.0 (trained on 6,234 songs)
# Current: hihat-v2.0.0 (trained on 4,891 songs)
# Update? [y/n]: y
# ✓ Downloaded and validated hihat-v2.1.0
# ✓ Updated root midiconfig.yaml with new presets
```

Or automatic updates (opt-in):
```yaml
# midiconfig.yaml
auto_update_models: true
check_interval_days: 7
```

## What Gets Shared

### ✓ Included (Anonymous)
- **Spectral features**: BodyE, SizzleE, GeoMean, etc.
- **Labels**: Open vs closed classification
- **Metadata**: Tempo, genre (optional), sample count
- **Quality metrics**: How well rules performed

### ✗ Excluded (Private)
- **Audio files**: Never uploaded
- **Filenames**: No song/artist identification
- **Timestamps**: No timing information
- **User identity**: Completely anonymous
- **Project structure**: No file paths

### Example: What Can't Be Reconstructed

**Original context**:
```
Project: "4 - Taylor Swift - Ruin The Friendship"
File: /Users/you/music/taylor_swift_reputation/06_ruin.wav
Open hi-hat at: 1.962s, 7.755s, 13.944s
```

**Submitted data**:
```json
{
  "features": [
    {"GeoMean": 906.2, "SustainMs": 181.6, "OpenHH": 1},
    {"GeoMean": 823.2, "SustainMs": 183.8, "OpenHH": 1}
  ],
  "metadata": {"tempo": 124, "genre": "pop"}
}
```

**No way to determine**: Song, artist, album, timing, or any identifying info ✓

## Privacy & Security

### Design Principles

1. **Minimal Data**: Only features needed for ML, nothing more
2. **Local Processing**: All audio analysis happens on user's machine
3. **Anonymous by Design**: Can't be de-anonymized even by us
4. **Opt-in Only**: Never automatic without consent
5. **Transparent**: Open source training code, auditable

### Technical Safeguards

```python
def anonymize_training_data(features_df, metadata):
    """
    Remove all potentially identifying information.
    """
    # Drop any timing columns
    features_df = features_df.drop(columns=['Time', 'timestamp'], errors='ignore')
    
    # Shuffle rows (removes temporal ordering)
    features_df = features_df.sample(frac=1.0).reset_index(drop=True)
    
    # Hash any IDs
    if 'project_id' in metadata:
        metadata['project_id'] = hashlib.sha256(
            metadata['project_id'].encode()
        ).hexdigest()[:8]
    
    # Remove filename references
    metadata.pop('filename', None)
    metadata.pop('filepath', None)
    metadata.pop('project_name', None)
    
    # Quantize metadata to reduce uniqueness
    if 'tempo' in metadata:
        metadata['tempo'] = round(metadata['tempo'] / 5) * 5  # Round to nearest 5 BPM
    
    return features_df, metadata
```

### User Controls

```yaml
# .larsnet-privacy (user home directory)
data_sharing:
  enabled: false              # Opt-in required
  allow_metadata: true        # Include tempo/genre
  allow_quality_metrics: true # Share how well rules performed
  anonymous_id: null          # Generated on first opt-in
  
# User can:
- Review every submission before it's sent
- Disable at any time
- Delete their contributions
- See aggregated statistics
```

## Universal Model Benefits

### Immediate Benefits
- **Better out-of-box accuracy**: New users get 95%+ without any tuning
- **Rare pattern detection**: Learn ghost notes, flams, half-open techniques
- **Genre adaptation**: Models trained on jazz, metal, pop, etc.
- **Robust to recording quality**: Learned from studio + home recordings

### Long-term Benefits
- **Continuous improvement**: Model gets better every week
- **Cross-instrument learning**: Hi-hat patterns inform cymbal detection
- **Transfer learning**: New instruments bootstrap from existing knowledge
- **Community-driven**: Users vote on best model versions

## Incentives for Contributing

### Recognition
- **Leaderboard**: Top contributors by sample count
- **Badges**: "Data Contributor", "Model Trainer", "Validation Helper"
- **Credits**: Contributors named in release notes (optional)

### Early Access
- **Beta features**: Contributors test new detection algorithms first
- **Priority support**: Faster response on forum/Discord
- **Feature requests**: Contributors influence roadmap

### Shared Success
- **Better for everyone**: Your data helps the next user
- **Specialized models**: Enough metal samples → metal-specific model
- **Research contributions**: Academic papers citing community data

## Implementation Phases

### Phase 1: Local Export (Immediate)
- ✓ Export training data to JSON
- ✓ Anonymization safeguards
- ✓ Local review interface

### Phase 2: Manual Submission (1-2 months)
- Web form for uploading JSON files
- Automatic validation
- Thank you page with stats

### Phase 3: Aggregation Backend (2-3 months)
- Secure storage (encrypted at rest)
- Training pipeline (weekly runs)
- Model versioning & distribution

### Phase 4: Automatic Updates (3-4 months)
- Model registry API
- Client-side update checker
- Validation & rollback system

### Phase 5: Advanced Features (4-6 months)
- Genre-specific models
- Active learning (system requests labels for uncertain cases)
- A/B testing (compare model versions)
- Real-time training (models update daily)

## API Design (Future)

### Submit Training Data
```http
POST https://api.larsnet-midi.com/v1/training-data
Content-Type: application/json
Authorization: Bearer <anonymous_token>

{
  "version": "1.0",
  "instrument": "hihat",
  "features": [...],
  "metadata": {...},
  "consent": {
    "terms_accepted": true,
    "allow_research_use": true
  }
}

Response:
{
  "id": "sub_abc123xyz",
  "status": "accepted",
  "contribution_count": 42,
  "next_model_eta": "2025-11-07T00:00:00Z"
}
```

### Download Model
```http
GET https://api.larsnet-midi.com/v1/models/hihat/latest

Response:
{
  "version": "2.1.0",
  "released": "2025-11-01T00:00:00Z",
  "training_stats": {
    "sample_count": 6234,
    "song_count": 1205,
    "contributor_count": 423
  },
  "performance": {
    "accuracy": 0.987,
    "precision": 0.983,
    "recall": 0.992,
    "f1_score": 0.987
  },
  "rules": {
    "simple": {...},
    "balanced": {...},
    "robust": {...}
  },
  "download_url": "https://cdn.larsnet-midi.com/models/hihat-v2.1.0.json",
  "checksum": "sha256:abc123..."
}
```

### Query Statistics
```http
GET https://api.larsnet-midi.com/v1/stats

Response:
{
  "total_contributions": 6234,
  "total_contributors": 423,
  "instruments": {
    "hihat": {"samples": 6234, "accuracy": 0.987},
    "snare": {"samples": 4891, "accuracy": 0.982},
    "kick": {"samples": 5102, "accuracy": 0.995}
  },
  "genres": {
    "pop": 2341,
    "rock": 1876,
    "metal": 892,
    "jazz": 634,
    "electronic": 491
  }
}
```

## Validation & Quality Control

### Submission Validation
- **Format check**: Valid JSON structure
- **Range check**: Features within expected bounds
- **Label consistency**: At least 2 examples of each class
- **Metadata validation**: Tempo, genre from allowed lists

### Quality Scoring
Each submission gets scored:
- **Diversity**: How different from existing data
- **Completeness**: Full feature set provided
- **Balance**: Good ratio of positive/negative examples
- **Usefulness**: Fills gaps in current model

High-quality submissions weighted more in training.

### Outlier Detection
- **Statistical outliers**: Features >3 standard deviations flagged
- **Adversarial detection**: Attempts to poison model detected
- **Duplicate detection**: Same features submitted multiple times

### Community Moderation
- **Validation set**: Random subset manually verified
- **Voting system**: Contributors can flag suspicious data
- **Reputation**: Repeat contributors build trust score

## Open Source Commitment

### Transparent Training
```
github.com/larsnet-midi/universal-models
├── training/
│   ├── pipeline.py         # Training script
│   ├── validation.py       # Quality checks
│   └── requirements.txt    # Dependencies
├── models/
│   ├── hihat-v2.1.0.json   # Published model
│   └── metadata.json       # Training stats
└── data/
    └── aggregated/          # Anonymized training data
        └── hihat_samples_anonymized.csv
```

All code open source, training data published (anonymized), models auditable.

### Research Use
- Academic researchers can access full dataset
- Must sign data use agreement
- Results published openly
- Contributions cited

## FAQ

**Q: Can my audio be identified from this data?**  
A: No. Only numerical features are shared, no audio, no timing, no filenames.

**Q: What if I change my mind?**  
A: You can delete your contributions at any time. We'll retrain without your data.

**Q: Who has access to the data?**  
A: Only the training pipeline. Aggregate statistics are public, raw data is not.

**Q: Can companies use this data?**  
A: Yes, under open source license. Everyone benefits equally.

**Q: What if someone submits bad data?**  
A: We have validation, outlier detection, and community moderation.

**Q: Will this make the local optimizer obsolete?**  
A: No! Universal model provides great defaults, local optimizer fine-tunes for your specific needs.

**Q: How do I know the model is actually better?**  
A: Published validation metrics, open source code, and you can test before updating.

## Get Involved

### As a Contributor
1. Use the system on your music
2. Export training data when you're happy with results
3. Submit to help others

### As a Validator
1. Review flagged submissions
2. Test new model versions
3. Report issues

### As a Developer
1. Improve anonymization techniques
2. Build better training pipelines
3. Create visualization tools

### As a Researcher
1. Analyze aggregate patterns
2. Publish papers on drum detection
3. Cite the community dataset

## Contact

- **Forum**: https://larsnet-midi.com/forum
- **Discord**: #training-data channel
- **Email**: privacy@larsnet-midi.com
- **GitHub**: github.com/larsnet-midi/universal-models

---

**This is a vision document**. Implementation pending community feedback and resource availability. All privacy and security measures will be thoroughly reviewed before launch.

**Last Updated**: 2025-10-31
