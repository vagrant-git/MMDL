import matplotlib
import matplotlib.pyplot as plt

print([f.name for f in matplotlib.font_manager.fontManager.ttflist if 'Times' in f.name])