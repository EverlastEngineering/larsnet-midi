import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Load data
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'data.csv')

# Label the open hi-hat events
open_hihat_times = [1.962, 7.755, 13.944, 19.574, 25.960, 31.753]
df['OpenHH'] = df['Time'].round(3).isin(open_hihat_times).astype(int)

# Define features and target
features = ['Str', 'Amp', 'BodyE', 'SizzleE', 'Total', 'GeoMean', 'SustainMs']
X = df[features]
y = df['OpenHH']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importance
importances = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])
print('\nFeature importances:')
for f, imp in importances:
    print(f'{f:10s} {imp:.3f}')

# Evaluate
print('\nRandom forest score:', rf.score(X_test, y_test))

# Train a shallow decision tree for interpretable rule extraction
tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
tree.fit(X, y)

# Display human-readable rules
rules = export_text(tree, feature_names=features)
print('\nInterpretable decision rules:\n')
print(rules)

# Identify likely open hi-hat candidates
df['Predicted_OpenHH'] = tree.predict(X)
print('\nPredicted open hi-hat times:')
print(df[df['Predicted_OpenHH'] == 1]['Time'].to_list())

# ============================================================================
# ANALYZE COMBINATION RULES WITH BETTER MARGINS
# ============================================================================

print('\n' + '='*70)
print('ANALYZING YOUR PROPOSED RULE: SustainMs > 150 AND GeoMean > 300')
print('='*70)

# Get stats for open hi-hats
open_hh = df[df['OpenHH'] == 1]
closed_hh = df[df['OpenHH'] == 0]

print('\nOpen Hi-Hat Statistics:')
print(f"  Count: {len(open_hh)}")
print(f"  GeoMean:   min={open_hh['GeoMean'].min():.1f}, max={open_hh['GeoMean'].max():.1f}, mean={open_hh['GeoMean'].mean():.1f}")
print(f"  SustainMs: min={open_hh['SustainMs'].min():.1f}, max={open_hh['SustainMs'].max():.1f}, mean={open_hh['SustainMs'].mean():.1f}")
print(f"  BodyE:     min={open_hh['BodyE'].min():.1f}, max={open_hh['BodyE'].max():.1f}, mean={open_hh['BodyE'].mean():.1f}")

print('\nClosed Hi-Hat Statistics (for comparison):')
print(f"  Count: {len(closed_hh)}")
print(f"  GeoMean:   min={closed_hh['GeoMean'].min():.1f}, max={closed_hh['GeoMean'].max():.1f}, mean={closed_hh['GeoMean'].mean():.1f}")
print(f"  SustainMs: min={closed_hh['SustainMs'].min():.1f}, max={closed_hh['SustainMs'].max():.1f}, mean={closed_hh['SustainMs'].mean():.1f}")
print(f"  BodyE:     min={closed_hh['BodyE'].min():.1f}, max={closed_hh['BodyE'].max():.1f}, mean={closed_hh['BodyE'].mean():.1f}")

print('\n' + '-'*70)
print('TESTING VARIOUS COMBINATION RULES:')
print('-'*70)

# Test various combination rules
test_rules = [
    ('GeoMean > 535', lambda row: row['GeoMean'] > 535),
    ('SustainMs > 150 AND GeoMean > 300', lambda row: row['SustainMs'] > 150 and row['GeoMean'] > 300),
    ('SustainMs > 140 AND GeoMean > 400', lambda row: row['SustainMs'] > 140 and row['GeoMean'] > 400),
    ('SustainMs > 150 AND GeoMean > 500', lambda row: row['SustainMs'] > 150 and row['GeoMean'] > 500),
    ('SustainMs > 90 AND GeoMean > 500', lambda row: row['SustainMs'] > 90 and row['GeoMean'] > 500),
    ('SustainMs > 100 AND BodyE > 100 AND GeoMean > 400', lambda row: row['SustainMs'] > 100 and row['BodyE'] > 100 and row['GeoMean'] > 400),
    ('SustainMs > 150 AND BodyE > 80', lambda row: row['SustainMs'] > 150 and row['BodyE'] > 80),
]

for rule_name, rule_func in test_rules:
    df['test_pred'] = df.apply(rule_func, axis=1).astype(int)
    
    # Count true positives, false positives, false negatives
    tp = ((df['OpenHH'] == 1) & (df['test_pred'] == 1)).sum()
    fp = ((df['OpenHH'] == 0) & (df['test_pred'] == 1)).sum()
    fn = ((df['OpenHH'] == 1) & (df['test_pred'] == 0)).sum()
    tn = ((df['OpenHH'] == 0) & (df['test_pred'] == 0)).sum()
    
    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f'\nRule: {rule_name}')
    print(f'  Catches {tp}/6 open hi-hats ({recall*100:.0f}% recall)')
    print(f'  False positives: {fp} closed hits misclassified as open')
    print(f'  Precision: {precision*100:.1f}% (of detected "opens", how many are real)')
    
    if fn > 0:
        missed = df[(df['OpenHH'] == 1) & (df['test_pred'] == 0)]
        print(f'  MISSED open hi-hats at times: {missed["Time"].tolist()}')

print('\n' + '='*70)
print('RECOMMENDATION:')
print('='*70)
print('Look for rules with:')
print('  - 100% recall (catches all 6 open hi-hats)')
print('  - High precision (few false positives)')
print('  - Safe margins from the minimum values')