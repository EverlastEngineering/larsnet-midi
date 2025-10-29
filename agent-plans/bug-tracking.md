# Bug Tracking

## Bug: MPS batch_size > 1 slower than batch_size=1
- **Status**: Fixed
- **Priority**: High
- **Description**: MPS device performs worse with higher batch sizes even at low overlap
- **Expected Behavior**: Higher batch sizes should improve performance
- **Actual Behavior**: batch_size=1 consistently faster than 2 or 4 across all overlap values
- **Fixed in Commit**: Changed MPS batch size logic to always use batch_size=1
- **Root Cause**: MPS unified memory architecture has different parallelization characteristics
