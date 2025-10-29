# Bug Tracking

## Bug: Device dropdown missing Auto option
- **Status**: Fixed
- **Priority**: Medium
- **Description**: Separation settings UI only offered CPU/CUDA/MPS, no auto-detection
- **Expected Behavior**: Should have "Auto (Recommended)" as default option
- **Actual Behavior**: Defaulted to CPU, requiring manual device selection
- **Fixed in Commit**: Added Auto option with detect_best_device() backend support

## Bug: MDX23C model file not in Git LFS
- **Status**: Fixed
- **Priority**: High
- **Description**: 417MB .ckpt file tracked in normal git instead of Git LFS
- **Expected Behavior**: Large model files should use Git LFS for efficient storage
- **Actual Behavior**: mdx_models/drumsep_5stems_mdx23c_jarredou.ckpt tracked as regular file
- **Fixed in Commit**: Added mdx_models/**/*.ckpt to .gitattributes for LFS tracking
- **Note**: File needs to be migrated to LFS: `git lfs migrate import --include="mdx_models/**/*.ckpt"`


