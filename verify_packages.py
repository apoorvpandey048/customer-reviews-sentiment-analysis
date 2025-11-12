"""Verify all required packages for the notebook are installed"""
import sys

print("Checking installed packages for EDA notebook...\n")
print("="*60)

packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'matplotlib': 'Plotting library',
    'seaborn': 'Statistical visualization',
    'plotly': 'Interactive plots',
    'wordcloud': 'Word cloud generation',
    'torch': 'PyTorch deep learning',
    'transformers': 'HuggingFace transformers',
    'datasets': 'HuggingFace datasets',
    'sklearn': 'Scikit-learn ML library',
    'scipy': 'Scientific computing',
    'nltk': 'Natural language toolkit',
}

success = []
failed = []

for package, description in packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        success.append((package, version, description))
        print(f"✓ {package:15} v{version:10} - {description}")
    except ImportError as e:
        failed.append((package, description))
        print(f"✗ {package:15} {'':10} - {description} [MISSING]")

print("="*60)
print(f"\n✓ Successfully installed: {len(success)}/{len(packages)} packages")

if failed:
    print(f"\n✗ Missing packages: {len(failed)}")
    for pkg, desc in failed:
        print(f"  - {pkg}: {desc}")
    print("\nInstall missing packages with:")
    print(f"pip install {' '.join([p[0] for p in failed])}")
else:
    print("\n🎉 All required packages are installed!")
    print("✓ The notebook is ready to run!")
