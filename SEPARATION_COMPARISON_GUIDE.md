# Separation Configuration Comparison Guide

When working with troublesome audio files, different separation configurations can produce significantly different results. This guide explains how to systematically compare multiple configurations to find the best settings for your audio.

## Quick Start

```bash
# Compare default configurations for a specific project
python compare_separation_configs.py 1

# Use GPU for faster processing
python compare_separation_configs.py 1 --device cuda

# Use custom configurations
python compare_separation_configs.py 1 --config my_configs.yaml
```

## What Gets Compared

The script processes your audio with multiple combinations of:

1. **Wiener Filter** - Post-processing that reduces artifacts and improves separation
2. **EQ Cleanup** - Frequency-specific filtering to reduce bleed between stems

### Default Configurations

The script tests 7 configurations by default:

| Configuration | Wiener Filter | EQ | Best For |
|--------------|---------------|-----|----------|
| `baseline` | Disabled | No | Reference - raw separation |
| `wiener-2.0` | α=2.0 | No | Strong artifact reduction |
| `wiener-1.0` | α=1.0 | No | Moderate artifact reduction |
| `wiener-0.4` | α=0.4 | No | Gentle artifact reduction |
| `eq` | Disabled | Yes | Frequency-specific bleed reduction |
| `wiener-2.0_eq` | α=2.0 | Yes | Maximum cleanup |
| `wiener-1.0_eq` | α=1.0 | Yes | Balanced cleanup |

## Output Structure

All comparison results are saved to `project_dir/for_comparison/`:

```
project_dir/
├── original_audio.wav
├── for_comparison/
│   ├── baseline/
│   │   └── original_audio/
│   │       ├── original_audio-kick.wav
│   │       ├── original_audio-snare.wav
│   │       └── ...
│   ├── wiener-2.0/
│   │   └── original_audio/
│   │       └── ...
│   ├── eq/
│   │   └── original_audio/
│   │       └── ...
│   └── ...
```

## How to Compare

### 1. Generate Comparisons

```bash
# For project #1
python compare_separation_configs.py 1
```

### 2. Listen and Evaluate

Load the stems from each configuration folder into your DAW:

- **Solo each stem** - Check for clarity and minimal bleed
- **Listen to problem areas** - Focus on sections that were previously troublesome
- **Check frequency ranges**:
  - Low end: Kick should be clean without cymbal bleed
  - Midrange: Snare should be clear without kick rumble
  - High end: Hihat/cymbals should be crisp without low-frequency contamination

### 3. Choose Your Winner

Once you identify the best configuration:

```bash
# Process normally with those settings
python separate.py 1 --wiener 2.0 --eq
```

Or copy the stems from `for_comparison/<best_config>/` to your project's `stems/` directory.

## Custom Configurations

Create a custom YAML file to test specific parameter combinations:

```yaml
# my_configs.yaml
configs:
  - name: "gentle"
    wiener: 0.5
    eq: true
  
  - name: "aggressive"
    wiener: 3.0
    eq: true
  
  - name: "eq_heavy"
    wiener: null
    eq: true
```

Then run:

```bash
python compare_separation_configs.py 1 --config my_configs.yaml
```

See `comparison_configs.example.yaml` for more examples.

## Command-Line Options

```
python compare_separation_configs.py [project_number] [options]

Arguments:
  project_number    Project number to process (optional, will prompt if omitted)

Options:
  -d, --device      Device to use: 'cpu' or 'cuda' (default: cpu)
  -c, --config      Path to custom configuration YAML file
  --cleanup         Remove original stems directory before processing
  -h, --help        Show help message
```

## Performance Tips

### Use GPU When Available

GPU processing is significantly faster for comparing multiple configurations:

```bash
python compare_separation_configs.py 1 --device cuda
```

### Start with Fewer Configs

If testing many configurations, start with a subset to quickly identify promising approaches:

```yaml
# quick_test.yaml
configs:
  - name: "baseline"
    wiener: null
    eq: false
  - name: "moderate"
    wiener: 1.0
    eq: true
  - name: "aggressive"
    wiener: 2.0
    eq: true
```

### Clean Up After Testing

Once you've identified the best configuration, you may want to remove the comparison directory to save disk space:

```bash
rm -rf project_dir/for_comparison/
```

## Understanding the Results

### Wiener Filter Effects

- **Low values (0.3-0.7)**: Gentle processing, preserves dynamics
- **Medium values (1.0-1.5)**: Balanced artifact reduction
- **High values (2.0+)**: Aggressive artifact reduction, may affect transients

### EQ Effects

The EQ settings are configured in `eq.yaml` and apply frequency-specific filtering:

- **Kick**: Removes sub-bass rumble and high-frequency cymbal bleed
- **Snare**: Removes low-frequency kick bleed
- **Toms**: Removes extreme low and high frequencies
- **Hihat**: High-pass filter for clarity
- **Cymbals**: Focused on high-frequency content

You can customize these settings per-project by creating a project-specific `eq.yaml`.

## Troubleshooting

### "No audio files found in project directory"

Ensure your project has an audio file (`.wav`, `.mp3`, `.flac`) in its root directory.

### "Config file not found"

Check that your custom config file path is correct and the file exists.

### Comparison is slow

- Use `--device cuda` if you have a compatible GPU
- Reduce the number of configurations in your custom config file
- Process shorter audio clips first to test configurations

### All configurations sound similar

Some audio doesn't benefit much from post-processing. Try:
- Adjusting EQ settings in `eq.yaml`
- Testing more extreme Wiener filter values
- Checking if the original separation quality is already good

## Related Documentation

- `README.md` - Main project documentation
- `separate.py` - Standard separation script
- `eq.yaml` - EQ configuration settings
- `config.yaml` - LarsNet model configuration
